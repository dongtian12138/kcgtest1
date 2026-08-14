#include <openssl/evp.h>

#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/rdf_loader/rdf_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr char kAuditSchema[] = "kcg_moveit_connector_pick_collision_audit_v1";
constexpr char kReportSchema[] = "kcg_d38999_closure_clearance_analysis_v1";

struct Arguments
{
  fs::path project_root;
  fs::path config;
  fs::path output;
  bool has_output{ false };
  bool report_only{ false };
};

struct Input
{
  fs::path relative_path;
  fs::path absolute_path;
  std::string expected_hash;
  std::string actual_hash;
};

struct DistanceSample
{
  bool available{ false };
  double distance{ std::numeric_limits<double>::max() };
  std::array<std::string, 2> pair;
  std::array<Eigen::Vector3d, 2> points{ Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero() };
};

std::string usage()
{
  return "usage: d38999_closure_clearance_analyzer --project-root PATH --config PATH "
         "[--output PATH] [--report-only]";
}

Arguments parseArguments(const int argc, char** argv)
{
  Arguments result;
  for (int index = 1; index < argc; ++index)
  {
    const std::string token(argv[index]);
    if (token == "--project-root" || token == "--config" || token == "--output")
    {
      if (index + 1 >= argc)
        throw std::runtime_error("missing value after " + token + "; " + usage());
      const fs::path value(argv[++index]);
      if (token == "--project-root")
        result.project_root = value;
      else if (token == "--config")
        result.config = value;
      else
      {
        result.output = value;
        result.has_output = true;
      }
    }
    else if (token == "--report-only")
      result.report_only = true;
    else if (token == "--help" || token == "-h")
    {
      std::cout << usage() << '\n';
      std::exit(0);
    }
    else
      throw std::runtime_error("unknown argument: " + token + "; " + usage());
  }
  if (result.project_root.empty() || result.config.empty())
    throw std::runtime_error("--project-root and --config are required; " + usage());
  result.project_root = fs::canonical(fs::absolute(result.project_root));
  if (!result.config.is_absolute())
    result.config = result.project_root / result.config;
  result.config = fs::canonical(result.config);
  if (result.has_output && !result.output.is_absolute())
    result.output = result.project_root / result.output;
  return result;
}

bool withinRoot(const fs::path& candidate, const fs::path& root)
{
  const fs::path relative = candidate.lexically_relative(root);
  return !relative.empty() && *relative.begin() != "..";
}

std::string sha256File(const fs::path& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open for SHA-256: " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("cannot initialize SHA-256");
  std::array<char, 65536> buffer{};
  while (stream)
  {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = stream.gcount();
    if (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(count)) != 1)
      throw std::runtime_error("SHA-256 update failed: " + path.string());
  }
  if (!stream.eof())
    throw std::runtime_error("SHA-256 input read failed: " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != 32U)
    throw std::runtime_error("SHA-256 finalization failed: " + path.string());
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < size; ++index)
    output << std::setw(2) << static_cast<unsigned int>(digest[index]);
  return output.str();
}

std::string readText(const fs::path& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open input: " + path.string());
  std::ostringstream result;
  result << stream.rdbuf();
  return result.str();
}

YAML::Node required(const YAML::Node& parent, const std::string& key)
{
  const YAML::Node result = parent[key];
  if (!result)
    throw std::runtime_error("missing YAML key: " + key);
  return result;
}

double finiteValue(const YAML::Node& node, const std::string& name)
{
  const double result = node.as<double>();
  if (!std::isfinite(result))
    throw std::runtime_error(name + " must be finite");
  return result;
}

std::vector<double> vectorOf(const YAML::Node& node, const std::size_t size, const std::string& name)
{
  if (!node.IsSequence() || node.size() != size)
    throw std::runtime_error(name + " must contain exactly " + std::to_string(size) + " entries");
  std::vector<double> result;
  result.reserve(size);
  for (std::size_t index = 0; index < size; ++index)
    result.push_back(finiteValue(node[index], name + "[" + std::to_string(index) + "]"));
  return result;
}

Eigen::Vector3d vector3Of(const YAML::Node& node, const std::string& name)
{
  const std::vector<double> values = vectorOf(node, 3, name);
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

Eigen::Isometry3d translated(const Eigen::Vector3d& translation)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = translation;
  return result;
}

double minimumJerk(const double fraction)
{
  return fraction * fraction * fraction * (10.0 + fraction * (-15.0 + 6.0 * fraction));
}

double inverseMinimumJerk(const double blend)
{
  double lower = 0.0;
  double upper = 1.0;
  for (std::size_t index = 0; index < 80U; ++index)
  {
    const double middle = 0.5 * (lower + upper);
    if (minimumJerk(middle) <= blend)
      lower = middle;
    else
      upper = middle;
  }
  return lower;
}

YAML::Node distanceNode(const DistanceSample& sample)
{
  YAML::Node node;
  node["available"] = sample.available;
  if (!sample.available)
    return node;
  node["signed_distance_m"] = sample.distance;
  node["pair"].push_back(sample.pair[0]);
  node["pair"].push_back(sample.pair[1]);
  for (const Eigen::Vector3d& point : sample.points)
  {
    YAML::Node point_node;
    point_node.push_back(point.x());
    point_node.push_back(point.y());
    point_node.push_back(point.z());
    node["nearest_points_world_m"].push_back(point_node);
  }
  return node;
}

DistanceSample convertDistance(const collision_detection::DistanceResult& result)
{
  DistanceSample sample;
  const auto& value = result.minimum_distance;
  if (!std::isfinite(value.distance))
    return sample;
  sample.available = true;
  sample.distance = value.distance;
  for (std::size_t index = 0; index < 2U; ++index)
  {
    sample.pair[index] = value.link_names[index];
    sample.points[index] = value.nearest_points[index];
  }
  return sample;
}

void writeReport(const YAML::Node& report, const Arguments& arguments)
{
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << report;
  if (!emitter.good())
    throw std::runtime_error("failed to serialize report");
  const std::string text = std::string(emitter.c_str()) + "\n";
  std::cout << text;
  if (arguments.has_output)
  {
    fs::create_directories(arguments.output.parent_path());
    std::ofstream stream(arguments.output);
    if (!stream)
      throw std::runtime_error("cannot open output: " + arguments.output.string());
    stream << text;
    if (!stream)
      throw std::runtime_error("cannot write output: " + arguments.output.string());
  }
}

int analyze(const Arguments& arguments)
{
  const YAML::Node config = YAML::LoadFile(arguments.config.string());
  if (required(config, "schema_version").as<std::string>() != kAuditSchema ||
      required(config, "profile").as<std::string>() != "d38999_shell25j_v1")
    throw std::runtime_error("analyzer requires the D38999 collision-audit v1 profile");

  const std::vector<std::string> required_names =
      { "urdf_xacro", "candidate_srdf", "d38999_pick", "d38999_scene", "d38999_proxy" };
  const YAML::Node raw_inputs = required(config, "inputs");
  if (!raw_inputs.IsMap() || raw_inputs.size() != required_names.size())
    throw std::runtime_error("D38999 audit input set does not match the contract");
  std::map<std::string, Input> inputs;
  for (const std::string& name : required_names)
  {
    const YAML::Node item = required(raw_inputs, name);
    Input input;
    input.relative_path = required(item, "path").as<std::string>();
    input.expected_hash = required(item, "sha256").as<std::string>();
    if (input.relative_path.is_absolute())
      throw std::runtime_error("input path must be project-relative: " + input.relative_path.string());
    input.absolute_path = fs::canonical(arguments.project_root / input.relative_path);
    if (!withinRoot(input.absolute_path, arguments.project_root) || !fs::is_regular_file(input.absolute_path))
      throw std::runtime_error("input escapes root or is not a file: " + input.relative_path.string());
    input.actual_hash = sha256File(input.absolute_path);
    if (input.expected_hash.size() != 64U || input.actual_hash != input.expected_hash)
      throw std::runtime_error("hash mismatch for " + name + ": expected=" + input.expected_hash +
                               " actual=" + input.actual_hash);
    inputs.emplace(name, std::move(input));
  }

  const YAML::Node pick = YAML::LoadFile(inputs.at("d38999_pick").absolute_path.string());
  const YAML::Node tabletop = YAML::LoadFile(inputs.at("d38999_scene").absolute_path.string());
  const YAML::Node proxy = YAML::LoadFile(inputs.at("d38999_proxy").absolute_path.string());
  if (required(pick, "schema_version").as<std::string>() != "kcg_d38999_tabletop_pick_v1" ||
      required(tabletop, "schema_version").as<std::string>() != "kcg_d38999_tabletop_scene_v1" ||
      required(proxy, "schema_version").as<std::string>() != "kcg_d38999_shell25j_proxy_v1")
    throw std::runtime_error("D38999 source schema mismatch");

  std::string urdf_xml;
  if (!rdf_loader::RDFLoader::loadXmlFileToString(urdf_xml, inputs.at("urdf_xacro").absolute_path.string(), {}))
    throw std::runtime_error("failed to expand URDF xacro");
  rdf_loader::RDFLoader rdf(urdf_xml, readText(inputs.at("candidate_srdf").absolute_path));
  if (!rdf.getURDF() || !rdf.getSRDF())
    throw std::runtime_error("failed to parse URDF/SRDF");
  auto model = std::make_shared<moveit::core::RobotModel>(rdf.getURDF(), rdf.getSRDF());
  planning_scene::PlanningScene scene(model);
  if (scene.getCollisionDetectorName() != "FCL")
    throw std::runtime_error("FCL is required for signed distance analysis");

  collision_detection::AllowedCollisionMatrix strict_acm = scene.getAllowedCollisionMatrix();
  std::size_t restored_never_count = 0;
  for (const srdf::Model::CollisionPair& pair : rdf.getSRDF()->getDisabledCollisionPairs())
  {
    if (pair.reason_ != "Never")
      continue;
    strict_acm.setEntry(pair.link1_, pair.link2_, false);
    ++restored_never_count;
  }
  if (restored_never_count != 76U)
    throw std::runtime_error("expected to restore exactly 76 reason=Never pairs");

  const YAML::Node motion = pick["motion"];
  const YAML::Node robot = pick["robot"];
  const std::vector<double> closure_arm =
      vectorOf(motion["closure_clearance_arm_rad"], 7U, "closure_clearance_arm_rad");
  const std::vector<double> open_hand = vectorOf(robot["open_hand_rad"], 4U, "open_hand_rad");
  const std::vector<double> grasp_hand = vectorOf(motion["grasp_hand_rad"], 4U, "grasp_hand_rad");
  std::vector<std::string> arm_names;
  std::vector<std::string> hand_names;
  for (const YAML::Node& item : robot["arm_joint_names"])
    arm_names.push_back(item.as<std::string>());
  for (const YAML::Node& item : robot["active_hand_joint_names"])
    hand_names.push_back(item.as<std::string>());
  if (arm_names.size() != 7U || hand_names.size() != 4U)
    throw std::runtime_error("unexpected arm/hand joint name count");

  const Eigen::Vector3d table_center = vector3Of(tabletop["table"]["center_m"], "table.center_m");
  const Eigen::Vector3d table_size = vector3Of(tabletop["table"]["size_m"], "table.size_m");
  const Eigen::Vector3d loose_initial =
      vector3Of(tabletop["loose_endpoint"]["initial_origin_m"], "initial_origin_m");
  const double bottom_offset =
      finiteValue(tabletop["loose_endpoint"]["body_bottom_offset_m"], "body_bottom_offset_m");
  const double table_top = table_center.z() + 0.5 * table_size.z();
  const Eigen::Vector3d settled_root(loose_initial.x(), loose_initial.y(), table_top - bottom_offset);

  const YAML::Node plug = proxy["proxy_geometry_m"]["plug"];
  const double overall_length = finiteValue(plug["overall_length"], "overall_length");
  const double rear_radius = finiteValue(plug["rear_body_radius"], "rear_body_radius");
  const double rear_length = finiteValue(plug["rear_body_length"], "rear_body_length");
  const double mating_radius = finiteValue(plug["mating_shell_outer_radius"], "mating_shell_outer_radius");
  const double mating_length = finiteValue(plug["mating_shell_length"], "mating_shell_length");
  const double nut_radius = finiteValue(plug["coupling_nut_outer_radius"], "coupling_nut_outer_radius");
  const double nut_length = finiteValue(plug["coupling_nut_length"], "coupling_nut_length");
  const std::string plug_id = "d38999_loose_endpoint";
  const std::vector<shapes::ShapeConstPtr> shapes = {
    std::make_shared<shapes::Cylinder>(rear_radius, rear_length),
    std::make_shared<shapes::Cylinder>(mating_radius, mating_length),
    std::make_shared<shapes::Cylinder>(nut_radius, nut_length)
  };
  const EigenSTL::vector_Isometry3d shape_poses = {
    translated(Eigen::Vector3d(0.0, 0.0, overall_length - 0.5 * rear_length)),
    translated(Eigen::Vector3d(0.0, 0.0, 0.5 * mating_length)),
    translated(Eigen::Vector3d(0.0, 0.0, 0.5 * overall_length))
  };
  scene.getWorldNonConst()->addToObject(plug_id, translated(settled_root), shapes, shape_poses);

  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  auto setBlend = [&](const double blend) {
    if (!std::isfinite(blend) || blend < 0.0 || blend > 1.0)
      throw std::runtime_error("closure blend is outside [0, 1]");
    for (std::size_t index = 0; index < arm_names.size(); ++index)
      state.setJointPositions(arm_names[index], &closure_arm[index]);
    for (std::size_t index = 0; index < hand_names.size(); ++index)
    {
      const double value = open_hand[index] + blend * (grasp_hand[index] - open_hand[index]);
      state.setJointPositions(hand_names[index], &value);
    }
    state.updateCollisionBodyTransforms();
  };

  auto selfDistance = [&](const double blend) {
    setBlend(blend);
    collision_detection::DistanceRequest request;
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::GLOBAL;
    request.acm = &strict_acm;
    request.enableGroup(model);
    collision_detection::DistanceResult result;
    scene.getCollisionEnvUnpadded()->distanceSelf(request, result, state);
    return convertDistance(result);
  };

  const double requested_clearance = 0.001;
  const std::size_t dense_intervals = 20000U;
  double last_safe_blend = 0.0;
  double first_unsafe_blend = 1.0;
  DistanceSample last_safe_sample = selfDistance(0.0);
  DistanceSample first_unsafe_sample;
  if (!last_safe_sample.available || last_safe_sample.distance < requested_clearance)
    throw std::runtime_error("closure begins below the requested 1 mm self-clearance");
  bool crossing_found = false;
  for (std::size_t index = 1; index <= dense_intervals; ++index)
  {
    const double blend = static_cast<double>(index) / static_cast<double>(dense_intervals);
    const DistanceSample sample = selfDistance(blend);
    if (!sample.available)
      throw std::runtime_error("FCL self-distance became unavailable during dense scan");
    if (sample.distance < requested_clearance)
    {
      first_unsafe_blend = blend;
      first_unsafe_sample = sample;
      crossing_found = true;
      break;
    }
    last_safe_blend = blend;
    last_safe_sample = sample;
  }
  if (!crossing_found)
  {
    last_safe_blend = 1.0;
    last_safe_sample = selfDistance(1.0);
    first_unsafe_blend = 1.0;
    first_unsafe_sample = last_safe_sample;
  }
  else
  {
    for (std::size_t iteration = 0; iteration < 80U; ++iteration)
    {
      const double middle = 0.5 * (last_safe_blend + first_unsafe_blend);
      const DistanceSample sample = selfDistance(middle);
      if (sample.distance >= requested_clearance)
      {
        last_safe_blend = middle;
        last_safe_sample = sample;
      }
      else
      {
        first_unsafe_blend = middle;
        first_unsafe_sample = sample;
      }
    }
  }

  const std::size_t closure_steps = static_cast<std::size_t>(std::llround(
      finiteValue(motion["closure_duration_s"], "closure_duration_s") *
      finiteValue(tabletop["physics"]["rate_hz"], "rate_hz")));
  std::size_t first_discrete_below_margin_step = 0U;
  DistanceSample first_discrete_below_margin;
  std::size_t first_discrete_collision_step = 0U;
  DistanceSample first_discrete_collision;
  for (std::size_t step = 1; step <= closure_steps; ++step)
  {
    const double fraction = static_cast<double>(step) / static_cast<double>(closure_steps);
    const DistanceSample sample = selfDistance(minimumJerk(fraction));
    if (first_discrete_below_margin_step == 0U && sample.distance < requested_clearance)
    {
      first_discrete_below_margin_step = step;
      first_discrete_below_margin = sample;
    }
    if (first_discrete_collision_step == 0U && sample.distance <= 0.0)
    {
      first_discrete_collision_step = step;
      first_discrete_collision = sample;
    }
  }

  const std::map<std::string, std::vector<std::string>> finger_groups = {
    { "finger_1", { "f1Link1", "f1Link2", "f1Link3" } },
    { "finger_2", { "f2Link1", "f2Link2" } },
    { "finger_3", { "f3Link1", "f3Link2", "f3Link3" } }
  };
  auto fingerDistance = [&](const double blend, const std::vector<std::string>& group_links) {
    setBlend(blend);
    collision_detection::AllowedCollisionMatrix isolated = strict_acm;
    for (const std::string& link : model->getLinkModelNamesWithCollisionGeometry())
      isolated.setEntry(link, plug_id, true);
    for (const std::string& link : group_links)
      isolated.setEntry(link, plug_id, false);
    collision_detection::DistanceRequest request;
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::GLOBAL;
    request.acm = &isolated;
    request.enableGroup(model);
    collision_detection::DistanceResult result;
    scene.getCollisionEnv()->distanceRobot(request, result, state);
    return convertDistance(result);
  };

  std::map<std::string, DistanceSample> finger_distances;
  std::map<std::string, double> finger_contact_onset_blends;
  std::map<std::string, DistanceSample> finger_contact_onset_samples;
  std::map<std::string, DistanceSample> b085_finger_distances;
  std::map<std::string, DistanceSample> b080_finger_distances;
  std::size_t touching_finger_count = 0U;
  const double b080_blend = 0.80;
  const double b085_blend = 0.85;
  double simultaneous_contact_onset_blend = 0.0;
  bool all_fingers_reach_contact_before_self_margin = true;
  for (const auto& group : finger_groups)
  {
    const DistanceSample sample = fingerDistance(last_safe_blend, group.second);
    if (!sample.available)
      throw std::runtime_error("plug distance unavailable for " + group.first);
    finger_distances.emplace(group.first, sample);
    if (sample.distance <= 0.0)
      ++touching_finger_count;

    const DistanceSample b085_sample = fingerDistance(b085_blend, group.second);
    if (!b085_sample.available)
      throw std::runtime_error("B085 plug distance unavailable for " + group.first);
    b085_finger_distances.emplace(group.first, b085_sample);
    const DistanceSample b080_sample = fingerDistance(b080_blend, group.second);
    if (!b080_sample.available)
      throw std::runtime_error("B080 plug distance unavailable for " + group.first);
    b080_finger_distances.emplace(group.first, b080_sample);

    double onset_lower = 0.0;
    double onset_upper = 0.0;
    DistanceSample lower_sample = fingerDistance(0.0, group.second);
    DistanceSample upper_sample = lower_sample;
    if (lower_sample.distance > 0.0)
    {
      constexpr std::size_t onset_intervals = 1000U;
      bool onset_found = false;
      for (std::size_t index = 1; index <= onset_intervals; ++index)
      {
        const double blend = last_safe_blend * static_cast<double>(index) / static_cast<double>(onset_intervals);
        const DistanceSample candidate = fingerDistance(blend, group.second);
        if (candidate.distance <= 0.0)
        {
          onset_upper = blend;
          upper_sample = candidate;
          onset_lower = last_safe_blend * static_cast<double>(index - 1U) /
                        static_cast<double>(onset_intervals);
          lower_sample = fingerDistance(onset_lower, group.second);
          onset_found = true;
          break;
        }
      }
      if (!onset_found)
      {
        all_fingers_reach_contact_before_self_margin = false;
        onset_lower = last_safe_blend;
        onset_upper = last_safe_blend;
        upper_sample = sample;
      }
      else
      {
        for (std::size_t iteration = 0; iteration < 60U; ++iteration)
        {
          const double middle = 0.5 * (onset_lower + onset_upper);
          const DistanceSample candidate = fingerDistance(middle, group.second);
          if (candidate.distance > 0.0)
          {
            onset_lower = middle;
            lower_sample = candidate;
          }
          else
          {
            onset_upper = middle;
            upper_sample = candidate;
          }
        }
      }
    }
    finger_contact_onset_blends.emplace(group.first, onset_upper);
    finger_contact_onset_samples.emplace(group.first, upper_sample);
    simultaneous_contact_onset_blend = std::max(simultaneous_contact_onset_blend, onset_upper);
  }

  std::map<std::string, DistanceSample> simultaneous_contact_distances;
  if (all_fingers_reach_contact_before_self_margin)
    for (const auto& group : finger_groups)
      simultaneous_contact_distances.emplace(group.first, fingerDistance(simultaneous_contact_onset_blend, group.second));

  std::vector<double> safe_hand(4U, 0.0);
  for (std::size_t index = 0; index < safe_hand.size(); ++index)
    safe_hand[index] = open_hand[index] + last_safe_blend * (grasp_hand[index] - open_hand[index]);
  const double safe_fraction = inverseMinimumJerk(last_safe_blend);
  const std::size_t last_discrete_step_not_exceeding_boundary =
      static_cast<std::size_t>(std::floor(safe_fraction * static_cast<double>(closure_steps) + 1.0e-12));
  const bool three_finger_contact = touching_finger_count == 3U;
  const DistanceSample b085_self_distance = selfDistance(b085_blend);
  const DistanceSample b080_self_distance = selfDistance(b080_blend);
  const bool b085_three_contact_reachable = std::all_of(
      b085_finger_distances.begin(), b085_finger_distances.end(),
      [](const auto& entry) { return entry.second.available && entry.second.distance <= 0.0; });
  const bool b085_meets_self_margin = b085_self_distance.available && b085_self_distance.distance >= requested_clearance;
  const bool b085_is_recommended_physics_candidate = b085_three_contact_reachable && b085_meets_self_margin;
  const bool b080_three_contact_reachable = std::all_of(
      b080_finger_distances.begin(), b080_finger_distances.end(),
      [](const auto& entry) { return entry.second.available && entry.second.distance <= 0.0; });
  const bool b080_meets_self_margin = b080_self_distance.available && b080_self_distance.distance >= requested_clearance;

  YAML::Node report;
  report["schema_version"] = kReportSchema;
  report["status"] = b085_is_recommended_physics_candidate ?
                         "ANALYSIS_COMPLETE_B085_RECOMMENDED_FOR_PHYSICS_AB" :
                         "ANALYSIS_COMPLETE_NO_SAFE_THREE_FINGER_PHYSICS_CANDIDATE";
  report["safe_grasp_feasible_under_static_proxy"] = false;
  report["three_finger_contact_reachable_before_self_margin"] =
      all_fingers_reach_contact_before_self_margin && three_finger_contact;
  report["static_feasibility_reason"] =
      "Negative finger-plug signed distance means rigid-shape interpenetration, not a realizable exact position. "
      "The command is only a PD/effort preload candidate whose physical equilibrium must be verified in PhysX.";
  report["continuous_collision_verified"] = false;
  report["continuous_collision_reason"] =
      "The result is a dense scalar scan plus local bisection, not a swept-volume continuous collision proof.";
  report["project_root"] = arguments.project_root.string();
  report["audit_config"] = arguments.config.string();
  for (const auto& entry : inputs)
  {
    report["inputs"][entry.first]["path"] = entry.second.relative_path.string();
    report["inputs"][entry.first]["expected_sha256"] = entry.second.expected_hash;
    report["inputs"][entry.first]["actual_sha256"] = entry.second.actual_hash;
    report["inputs"][entry.first]["hash_matches"] = entry.second.actual_hash == entry.second.expected_hash;
  }
  report["method"]["collision_detector"] = scene.getCollisionDetectorName();
  report["method"]["strict_reenabled_never_pair_count"] = restored_never_count;
  report["method"]["dense_blend_intervals"] = dense_intervals;
  report["method"]["bisection_iterations"] = 80U;
  report["method"]["required_signed_self_clearance_m"] = requested_clearance;
  report["method"]["closure_command_steps"] = closure_steps;
  report["safe_boundary"]["maximum_prefix_safe_blend"] = last_safe_blend;
  report["safe_boundary"]["first_unsafe_blend"] = first_unsafe_blend;
  report["safe_boundary"]["minimum_jerk_time_fraction"] = safe_fraction;
  report["safe_boundary"]["last_discrete_step_not_exceeding_boundary_one_based"] =
      last_discrete_step_not_exceeding_boundary;
  report["safe_boundary"]["strict_self_distance"] = distanceNode(last_safe_sample);
  report["safe_boundary"]["first_unsafe_strict_self_distance"] = distanceNode(first_unsafe_sample);
  for (std::size_t index = 0; index < hand_names.size(); ++index)
  {
    report["safe_boundary"]["active_hand_joint_names"].push_back(hand_names[index]);
    report["safe_boundary"]["active_hand_command_rad"].push_back(safe_hand[index]);
  }
  report["discrete_schedule"]["first_step_below_1mm_one_based"] = first_discrete_below_margin_step;
  report["discrete_schedule"]["first_step_below_1mm"] = distanceNode(first_discrete_below_margin);
  report["discrete_schedule"]["first_collision_step_one_based"] = first_discrete_collision_step;
  report["discrete_schedule"]["first_collision"] = distanceNode(first_discrete_collision);
  report["plug_proxy"]["nominal_outer_diameter_m"] = 2.0 * nut_radius;
  report["plug_proxy"]["geometry_kind"] = "conservative_compound_solid_cylinders";
  report["plug_proxy"]["ring_bores_filled"] = true;
  report["plug_proxy"]["settled_origin_world_m"].push_back(settled_root.x());
  report["plug_proxy"]["settled_origin_world_m"].push_back(settled_root.y());
  report["plug_proxy"]["settled_origin_world_m"].push_back(settled_root.z());
  report["plug_contact_at_safe_boundary"]["touch_definition"] = "signed_distance_m <= 0";
  report["plug_contact_at_safe_boundary"]["touching_finger_count"] = touching_finger_count;
  report["plug_contact_at_safe_boundary"]["three_finger_contact"] = three_finger_contact;
  for (const auto& entry : finger_distances)
  {
    report["plug_contact_at_safe_boundary"]["fingers"][entry.first] = distanceNode(entry.second);
    report["plug_contact_at_safe_boundary"]["fingers"][entry.first]["touching"] = entry.second.distance <= 0.0;
  }
  report["contact_onset"]["all_fingers_reach_contact_before_1mm_self_margin"] =
      all_fingers_reach_contact_before_self_margin;
  report["contact_onset"]["simultaneous_contact_reachability_blend"] = simultaneous_contact_onset_blend;
  for (const auto& group : finger_groups)
  {
    report["contact_onset"]["fingers"][group.first]["blend"] = finger_contact_onset_blends.at(group.first);
    report["contact_onset"]["fingers"][group.first]["distance_at_onset"] =
        distanceNode(finger_contact_onset_samples.at(group.first));
    if (all_fingers_reach_contact_before_self_margin)
      report["contact_onset"]["fingers"][group.first]["distance_at_simultaneous_onset"] =
          distanceNode(simultaneous_contact_distances.at(group.first));
  }
  report["recommended_physx_ab_candidate"]["name"] = "B085";
  report["recommended_physx_ab_candidate"]["closure_blend"] = b085_blend;
  report["recommended_physx_ab_candidate"]["active_hand_joint_names"] = hand_names;
  for (std::size_t index = 0; index < hand_names.size(); ++index)
    report["recommended_physx_ab_candidate"]["active_hand_command_rad"].push_back(
        open_hand[index] + b085_blend * (grasp_hand[index] - open_hand[index]));
  report["recommended_physx_ab_candidate"]["strict_self_distance"] = distanceNode(b085_self_distance);
  report["recommended_physx_ab_candidate"]["meets_1mm_self_margin"] = b085_meets_self_margin;
  report["recommended_physx_ab_candidate"]["three_finger_contact_reachable"] = b085_three_contact_reachable;
  report["recommended_physx_ab_candidate"]["recommended_for_physics_ab"] = b085_is_recommended_physics_candidate;
  report["recommended_physx_ab_candidate"]["per_finger_effort_limit_nm"] = 2.0;
  report["recommended_physx_ab_candidate"]["effort_limit_source"] = "user_confirmed_2026-08-12";
  for (const auto& entry : b085_finger_distances)
    report["recommended_physx_ab_candidate"]["finger_target_signed_distances"][entry.first] =
        distanceNode(entry.second);
  report["recommended_physx_ab_candidate"]["interpretation"] =
      "The negative target distances intentionally request preload beyond first rigid contact. PhysX must stop the "
      "fingers at contact and demonstrate finite torque-limited equilibrium without self-collision.";
  report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["name"] = "B080";
  report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["closure_blend"] = b080_blend;
  for (std::size_t index = 0; index < hand_names.size(); ++index)
    report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["active_hand_command_rad"].push_back(
        open_hand[index] + b080_blend * (grasp_hand[index] - open_hand[index]));
  report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["strict_self_distance"] =
      distanceNode(b080_self_distance);
  report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["meets_1mm_self_margin"] =
      b080_meets_self_margin;
  report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["three_finger_contact_reachable"] =
      b080_three_contact_reachable;
  for (const auto& entry : b080_finger_distances)
    report["recommended_physx_ab_candidate"]["lower_preload_a_candidate"]["finger_target_signed_distances"]
          [entry.first] = distanceNode(entry.second);
  report["limitations"]["physics_contact_verified"] = false;
  report["limitations"]["force_closure_verified"] = false;
  report["limitations"]["deformable_contact_or_compliance_modeled"] = false;
  report["limitations"]["flight_connector_cad_fidelity_verified"] = false;
  report["limitations"]["result_is_collision_free_continuous_path_proof"] = false;

  writeReport(report, arguments);
  return b085_is_recommended_physics_candidate || arguments.report_only ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    return analyze(parseArguments(argc, argv));
  }
  catch (const std::exception& error)
  {
    YAML::Node report;
    report["schema_version"] = kReportSchema;
    report["status"] = "FAIL_CLOSED_RUNTIME_ERROR";
    report["safe_grasp_feasible_under_static_proxy"] = false;
    report["error"] = error.what();
    YAML::Emitter emitter;
    emitter << report;
    std::cerr << emitter.c_str() << '\n';
    return 1;
  }
}
