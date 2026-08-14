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
#include <chrono>
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
constexpr char kConfigSchema[] = "kcg_d38999_nut_regrasp_search_v1";
constexpr char kReportSchema[] = "kcg_d38999_nut_regrasp_search_report_v1";

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

struct Anchor
{
  double tcp_z{ 0.0 };
  std::array<double, 7> arm{};
};

struct Candidate
{
  bool evaluated{ false };
  bool feasible{ false };
  double tcp_z{ 0.0 };
  double outer_blend{ 0.0 };
  double middle_blend{ 0.0 };
  std::array<double, 7> arm{};
  std::array<double, 4> hand{};
  double tcp_position_error{ 0.0 };
  double tcp_axis_error{ 0.0 };
  DistanceSample candidate_self;
  DistanceSample strict_self;
  DistanceSample environment;
  DistanceSample forbidden_endpoint;
  std::map<std::string, DistanceSample> nut;
  std::map<std::string, DistanceSample> body;
  double worst_nut_distance{ std::numeric_limits<double>::max() };
  double minimum_body_distance{ std::numeric_limits<double>::max() };
  double minimum_margin{ -std::numeric_limits<double>::max() };
  double maximum_nut_target_penetration{ std::numeric_limits<double>::max() };
};

struct PathMinimum
{
  bool available{ false };
  DistanceSample sample;
  std::string phase;
  std::size_t phase_step{ 0U };
  std::size_t global_step{ 0U };
};

struct PathAudit
{
  bool evaluated{ false };
  bool passed{ false };
  std::size_t sample_count{ 0U };
  std::size_t expected_sample_count{ 0U };
  std::size_t bounds_violation_count{ 0U };
  double maximum_joint_step{ 0.0 };
  PathMinimum candidate_self;
  PathMinimum strict_self;
  PathMinimum environment;
  PathMinimum nonfinger_endpoint;
  PathMinimum body_during_reapproach;
  PathMinimum nut_during_open_reposition;
};

std::string usage()
{
  return "usage: d38999_nut_regrasp_search --project-root PATH --config PATH "
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
  const std::vector<double> values = vectorOf(node, 3U, name);
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
    YAML::Node item;
    item.push_back(point.x());
    item.push_back(point.y());
    item.push_back(point.z());
    node["nearest_points_world_m"].push_back(item);
  }
  return node;
}

void updateMinimum(PathMinimum& minimum, const DistanceSample& sample, const std::string& phase,
                   const std::size_t phase_step, const std::size_t global_step)
{
  if (!sample.available)
    throw std::runtime_error("path distance sample is unavailable");
  if (!minimum.available || sample.distance < minimum.sample.distance)
  {
    minimum.available = true;
    minimum.sample = sample;
    minimum.phase = phase;
    minimum.phase_step = phase_step;
    minimum.global_step = global_step;
  }
}

YAML::Node pathMinimumNode(const PathMinimum& minimum)
{
  YAML::Node node;
  node["available"] = minimum.available;
  if (!minimum.available)
    return node;
  node["distance"] = distanceNode(minimum.sample);
  node["phase"] = minimum.phase;
  node["phase_step_one_based"] = minimum.phase_step;
  node["global_step_one_based"] = minimum.global_step;
  return node;
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

bool pairContains(const DistanceSample& sample, const std::string& name)
{
  return sample.available && (sample.pair[0] == name || sample.pair[1] == name);
}

YAML::Node candidateNode(const Candidate& candidate)
{
  YAML::Node node;
  node["evaluated"] = candidate.evaluated;
  if (!candidate.evaluated)
    return node;
  node["feasible"] = candidate.feasible;
  node["tcp_z_command_m"] = candidate.tcp_z;
  node["outer_finger_blend"] = candidate.outer_blend;
  node["middle_finger_blend"] = candidate.middle_blend;
  for (const double value : candidate.arm)
    node["arm_command_rad"].push_back(value);
  for (const double value : candidate.hand)
    node["hand_command_rad"].push_back(value);
  node["tcp_position_error_m"] = candidate.tcp_position_error;
  node["tcp_axis_error_rad"] = candidate.tcp_axis_error;
  node["minimum_acceptance_margin_m"] = candidate.minimum_margin;
  node["worst_nut_signed_distance_m"] = candidate.worst_nut_distance;
  node["minimum_body_signed_distance_m"] = candidate.minimum_body_distance;
  node["maximum_nut_target_penetration_m"] = candidate.maximum_nut_target_penetration;
  node["candidate_self_distance"] = distanceNode(candidate.candidate_self);
  node["strict_self_distance"] = distanceNode(candidate.strict_self);
  node["robot_environment_distance"] = distanceNode(candidate.environment);
  node["forbidden_endpoint_distance"] = distanceNode(candidate.forbidden_endpoint);
  for (const auto& entry : candidate.nut)
  {
    node["per_finger"][entry.first]["nut"] = distanceNode(entry.second);
    node["per_finger"][entry.first]["nut_touch"] = entry.second.available && entry.second.distance <= 0.0;
  }
  for (const auto& entry : candidate.body)
  {
    node["per_finger"][entry.first]["body"] = distanceNode(entry.second);
    node["per_finger"][entry.first]["body_clearance_at_least_1mm"] =
        entry.second.available && entry.second.distance >= 0.001;
  }
  return node;
}

int runSearch(const Arguments& arguments)
{
  const auto started = std::chrono::steady_clock::now();
  const YAML::Node config = YAML::LoadFile(arguments.config.string());
  if (required(config, "schema_version").as<std::string>() != kConfigSchema)
    throw std::runtime_error("nut regrasp search schema mismatch");
  if (required(config, "enabled").as<bool>())
    throw std::runtime_error("offline search contract must remain disabled");

  const std::vector<std::string> input_names =
      { "urdf_xacro", "candidate_srdf", "d38999_proxy", "d38999_scene", "d38999_assembly" };
  const YAML::Node raw_inputs = required(config, "inputs");
  if (!raw_inputs.IsMap() || raw_inputs.size() != input_names.size())
    throw std::runtime_error("nut regrasp input set does not match the exact contract");
  std::map<std::string, Input> inputs;
  for (const std::string& name : input_names)
  {
    const YAML::Node raw = required(raw_inputs, name);
    Input input;
    input.relative_path = required(raw, "path").as<std::string>();
    input.expected_hash = required(raw, "sha256").as<std::string>();
    if (input.relative_path.is_absolute())
      throw std::runtime_error("input path must be project-relative: " + input.relative_path.string());
    input.absolute_path = fs::canonical(arguments.project_root / input.relative_path);
    if (!withinRoot(input.absolute_path, arguments.project_root) || !fs::is_regular_file(input.absolute_path))
      throw std::runtime_error("input escapes root or is not a file: " + input.relative_path.string());
    input.actual_hash = sha256File(input.absolute_path);
    if (input.expected_hash.size() != 64U || input.expected_hash != input.actual_hash)
      throw std::runtime_error("hash mismatch for " + name + ": expected=" + input.expected_hash +
                               " actual=" + input.actual_hash);
    inputs.emplace(name, std::move(input));
  }

  const YAML::Node proxy = YAML::LoadFile(inputs.at("d38999_proxy").absolute_path.string());
  const YAML::Node tabletop = YAML::LoadFile(inputs.at("d38999_scene").absolute_path.string());
  const YAML::Node assembly = YAML::LoadFile(inputs.at("d38999_assembly").absolute_path.string());
  if (required(proxy, "schema_version").as<std::string>() != "kcg_d38999_shell25j_proxy_v1" ||
      required(tabletop, "schema_version").as<std::string>() != "kcg_d38999_tabletop_scene_v1" ||
      required(assembly, "schema_version").as<std::string>() != "kcg_d38999_assembly_baseline_v1")
    throw std::runtime_error("D38999 source schema mismatch");
  if (required(assembly, "enabled").as<bool>())
    throw std::runtime_error("assembly baseline unexpectedly enabled");

  std::string urdf_xml;
  if (!rdf_loader::RDFLoader::loadXmlFileToString(urdf_xml, inputs.at("urdf_xacro").absolute_path.string(), {}))
    throw std::runtime_error("failed to expand URDF xacro");
  rdf_loader::RDFLoader rdf(urdf_xml, readText(inputs.at("candidate_srdf").absolute_path));
  if (!rdf.getURDF() || !rdf.getSRDF())
    throw std::runtime_error("failed to parse URDF/SRDF");
  auto model = std::make_shared<moveit::core::RobotModel>(rdf.getURDF(), rdf.getSRDF());
  planning_scene::PlanningScene scene(model);
  if (scene.getCollisionDetectorName() != "FCL")
    throw std::runtime_error("FCL is required for signed distance search");

  collision_detection::AllowedCollisionMatrix candidate_acm = scene.getAllowedCollisionMatrix();
  collision_detection::AllowedCollisionMatrix strict_acm = candidate_acm;
  std::size_t restored_never_count = 0U;
  for (const srdf::Model::CollisionPair& pair : rdf.getSRDF()->getDisabledCollisionPairs())
  {
    if (pair.reason_ != "Never")
      continue;
    strict_acm.setEntry(pair.link1_, pair.link2_, false);
    ++restored_never_count;
  }
  const YAML::Node robot_config = required(config, "robot");
  if (restored_never_count != required(robot_config, "expected_never_pair_count").as<std::size_t>())
    throw std::runtime_error("unexpected reason=Never pair count");

  std::vector<std::string> arm_names;
  std::vector<std::string> hand_names;
  for (const YAML::Node& item : required(robot_config, "arm_joint_names"))
    arm_names.push_back(item.as<std::string>());
  for (const YAML::Node& item : required(robot_config, "active_hand_joint_names"))
    hand_names.push_back(item.as<std::string>());
  if (arm_names.size() != 7U || hand_names.size() != 4U)
    throw std::runtime_error("robot joint-name contract must contain seven arm and four active hand joints");
  for (const std::string& name : arm_names)
    if (!model->hasJointModel(name))
      throw std::runtime_error("arm joint absent from model: " + name);
  for (const std::string& name : hand_names)
    if (!model->hasJointModel(name))
      throw std::runtime_error("hand joint absent from model: " + name);
  const std::string tcp_link = required(robot_config, "grasp_tcp_link").as<std::string>();
  if (!model->hasLinkModel(tcp_link))
    throw std::runtime_error("grasp TCP link absent from model");
  const double fixed_q7 = finiteValue(required(robot_config, "fixed_q7_rad"), "fixed_q7_rad");

  const YAML::Node ik_family = required(config, "ik_family");
  const std::vector<double> tcp_xy = vectorOf(required(ik_family, "tcp_xy_m"), 2U, "tcp_xy_m");
  const double maximum_tcp_position_error =
      finiteValue(required(ik_family, "maximum_interpolated_tcp_position_error_m"),
                  "maximum_interpolated_tcp_position_error_m");
  const double maximum_tcp_axis_error =
      finiteValue(required(ik_family, "maximum_interpolated_tcp_axis_error_rad"),
                  "maximum_interpolated_tcp_axis_error_rad");
  std::vector<Anchor> anchors;
  for (const YAML::Node& raw : required(ik_family, "anchors"))
  {
    Anchor anchor;
    anchor.tcp_z = finiteValue(required(raw, "tcp_z_m"), "anchor.tcp_z_m");
    const std::vector<double> values = vectorOf(required(raw, "arm_rad"), 7U, "anchor.arm_rad");
    std::copy(values.begin(), values.end(), anchor.arm.begin());
    if (std::abs(anchor.arm[6] - fixed_q7) > 1.0e-12)
      throw std::runtime_error("IK anchor violates fixed q7 contract");
    if (!anchors.empty() && anchor.tcp_z <= anchors.back().tcp_z)
      throw std::runtime_error("IK anchors must be strictly increasing in TCP z");
    anchors.push_back(anchor);
  }
  if (anchors.size() < 2U)
    throw std::runtime_error("at least two IK anchors are required");

  auto armAt = [&](const double tcp_z) {
    if (tcp_z < anchors.front().tcp_z - 1.0e-12 || tcp_z > anchors.back().tcp_z + 1.0e-12)
      throw std::runtime_error("requested TCP z lies outside IK anchors");
    auto upper = std::upper_bound(anchors.begin(), anchors.end(), tcp_z,
                                  [](const double value, const Anchor& anchor) { return value < anchor.tcp_z; });
    if (upper == anchors.begin())
      return anchors.front().arm;
    if (upper == anchors.end())
      return anchors.back().arm;
    const Anchor& high = *upper;
    const Anchor& low = *(upper - 1);
    const double fraction = (tcp_z - low.tcp_z) / (high.tcp_z - low.tcp_z);
    std::array<double, 7> result{};
    for (std::size_t index = 0; index < result.size(); ++index)
      result[index] = low.arm[index] + fraction * (high.arm[index] - low.arm[index]);
    return result;
  };

  const YAML::Node plug = proxy["proxy_geometry_m"]["plug"];
  const double overall_length = finiteValue(required(plug, "overall_length"), "plug.overall_length");
  const double rear_radius = finiteValue(required(plug, "rear_body_radius"), "plug.rear_body_radius");
  const double rear_length = finiteValue(required(plug, "rear_body_length"), "plug.rear_body_length");
  const double mating_radius =
      finiteValue(required(plug, "mating_shell_outer_radius"), "plug.mating_shell_outer_radius");
  const double mating_length = finiteValue(required(plug, "mating_shell_length"), "plug.mating_shell_length");
  const double nut_radius =
      finiteValue(required(plug, "coupling_nut_outer_radius"), "plug.coupling_nut_outer_radius");
  const double nut_length = finiteValue(required(plug, "coupling_nut_length"), "plug.coupling_nut_length");

  const Eigen::Vector3d fixed_origin =
      vector3Of(assembly["datums"]["fixed"]["position_world_m"], "assembly.fixed.position_world_m");
  const Eigen::Vector3d fixed_axis =
      vector3Of(assembly["datums"]["fixed"]["axis_world"], "assembly.fixed.axis_world");
  if ((fixed_axis - Eigen::Vector3d::UnitZ()).norm() > 1.0e-12)
    throw std::runtime_error("v1 nut regrasp search requires fixed +Z assembly axis");
  const double engage_gap = finiteValue(assembly["axial_plan"]["engage_gap_m"], "assembly.engage_gap_m");
  const Eigen::Vector3d body_root = fixed_origin + engage_gap * fixed_axis;

  const Eigen::Vector3d table_center = vector3Of(tabletop["table"]["center_m"], "table.center_m");
  const Eigen::Vector3d table_size = vector3Of(tabletop["table"]["size_m"], "table.size_m");
  const Eigen::Vector3d fixture_center =
      vector3Of(tabletop["fixed_endpoint"]["fixture_center_m"], "fixture.center_m");
  const Eigen::Vector3d fixture_size =
      vector3Of(tabletop["fixed_endpoint"]["fixture_size_m"], "fixture.size_m");
  const YAML::Node receptacle = proxy["proxy_geometry_m"]["receptacle"];
  const double flange_side = finiteValue(required(receptacle, "flange_side"), "receptacle.flange_side");
  const double flange_thickness =
      finiteValue(required(receptacle, "flange_thickness"), "receptacle.flange_thickness");
  const double fixed_shell_radius =
      finiteValue(required(receptacle, "shell_outer_radius"), "receptacle.shell_outer_radius");
  const double fixed_shell_length =
      finiteValue(required(receptacle, "front_shell_length"), "receptacle.front_shell_length");
  const double fixed_rear_radius =
      finiteValue(required(receptacle, "rear_body_radius"), "receptacle.rear_body_radius");
  const double fixed_rear_length =
      finiteValue(required(receptacle, "rear_body_length"), "receptacle.rear_body_length");

  const std::string table_id = "d38999_regrasp_table";
  const std::string fixture_id = "d38999_regrasp_fixture";
  const std::string fixed_id = "d38999_regrasp_fixed_endpoint";
  const std::string body_id = "d38999_regrasp_body_assembly";
  const std::string nut_id = "d38999_regrasp_coupling_nut";
  scene.getWorldNonConst()->addToObject(
      table_id, translated(table_center), std::make_shared<shapes::Box>(table_size.x(), table_size.y(), table_size.z()),
      Eigen::Isometry3d::Identity());
  scene.getWorldNonConst()->addToObject(
      fixture_id, translated(fixture_center),
      std::make_shared<shapes::Box>(fixture_size.x(), fixture_size.y(), fixture_size.z()),
      Eigen::Isometry3d::Identity());
  const std::vector<shapes::ShapeConstPtr> fixed_shapes = {
    std::make_shared<shapes::Box>(flange_side, flange_side, flange_thickness),
    std::make_shared<shapes::Cylinder>(fixed_shell_radius, fixed_shell_length),
    std::make_shared<shapes::Cylinder>(fixed_rear_radius, fixed_rear_length)
  };
  const EigenSTL::vector_Isometry3d fixed_poses = {
    translated(Eigen::Vector3d(0.0, 0.0, -0.5 * flange_thickness)),
    translated(Eigen::Vector3d(0.0, 0.0, 0.5 * fixed_shell_length)),
    translated(Eigen::Vector3d(0.0, 0.0, -0.5 * fixed_rear_length))
  };
  scene.getWorldNonConst()->addToObject(fixed_id, translated(fixed_origin), fixed_shapes, fixed_poses);
  const std::vector<shapes::ShapeConstPtr> body_shapes = {
    std::make_shared<shapes::Cylinder>(rear_radius, rear_length),
    std::make_shared<shapes::Cylinder>(mating_radius, mating_length)
  };
  const EigenSTL::vector_Isometry3d body_poses = {
    translated(Eigen::Vector3d(0.0, 0.0, overall_length - 0.5 * rear_length)),
    translated(Eigen::Vector3d(0.0, 0.0, 0.5 * mating_length))
  };
  scene.getWorldNonConst()->addToObject(body_id, translated(body_root), body_shapes, body_poses);
  scene.getWorldNonConst()->addToObject(
      nut_id, translated(body_root), std::make_shared<shapes::Cylinder>(nut_radius, nut_length),
      translated(Eigen::Vector3d(0.0, 0.0, 0.5 * overall_length)));

  const std::vector<std::string> world_ids = { table_id, fixture_id, fixed_id, body_id, nut_id };
  const std::vector<std::string> environment_ids = { table_id, fixture_id, fixed_id };
  const std::map<std::string, std::vector<std::string>> finger_groups = {
    { "finger_1", { "f1Link1", "f1Link2", "f1Link3" } },
    { "finger_2", { "f2Link1", "f2Link2" } },
    { "finger_3", { "f3Link1", "f3Link2", "f3Link3" } }
  };
  std::set<std::string> all_finger_links;
  for (const auto& entry : finger_groups)
    all_finger_links.insert(entry.second.begin(), entry.second.end());

  const YAML::Node search = required(config, "search");
  const YAML::Node mapping = required(search, "hand_command_mapping");
  const double f1j1_fixed = finiteValue(required(mapping, "f1j1_fixed_rad"), "f1j1_fixed_rad");
  const double outer_scale =
      finiteValue(required(mapping, "f1j2_from_outer_blend_scale"), "outer blend scale");
  const double middle_scale =
      finiteValue(required(mapping, "f2j1_from_middle_blend_scale"), "middle blend scale");
  const double f3_scale =
      finiteValue(required(mapping, "f3j2_from_outer_blend_scale"), "f3 blend scale");
  if (std::abs(outer_scale - f3_scale) > 1.0e-12)
    throw std::runtime_error("v1 search requires symmetric outer-finger scaling");

  const YAML::Node acceptance = required(config, "acceptance");
  const double nut_touch_max =
      finiteValue(required(acceptance, "nut_touch_signed_distance_max_m"), "nut touch max");
  const double body_clearance_min =
      finiteValue(required(acceptance, "body_signed_clearance_min_m"), "body clearance min");
  const double candidate_self_min =
      finiteValue(required(acceptance, "candidate_self_clearance_min_m"), "candidate self min");
  const double strict_self_min =
      finiteValue(required(acceptance, "strict_self_clearance_min_m"), "strict self min");
  const double environment_min =
      finiteValue(required(acceptance, "robot_environment_clearance_min_m"), "environment min");
  if (nut_touch_max != 0.0 || body_clearance_min != 0.001 || candidate_self_min != 0.001 ||
      strict_self_min != 0.001 || environment_min != 0.005 ||
      !required(acceptance, "require_all_three_fingers_touch_nut").as<bool>() ||
      !required(acceptance, "require_all_three_fingers_clear_body").as<bool>() ||
      !required(acceptance, "require_continuous_collision_verified").as<bool>())
    throw std::runtime_error("nut regrasp acceptance thresholds are not canonical v1");

  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  std::size_t evaluated_count = 0U;
  std::size_t finite_count = 0U;
  double observed_maximum_tcp_position_error = 0.0;
  double observed_maximum_tcp_axis_error = 0.0;

  auto setState = [&](const double tcp_z, const double outer_blend, const double middle_blend,
                      Candidate& candidate) {
    candidate.arm = armAt(tcp_z);
    candidate.hand = { f1j1_fixed, outer_scale * outer_blend, middle_scale * middle_blend,
                       f3_scale * outer_blend };
    for (std::size_t index = 0; index < arm_names.size(); ++index)
      state.setJointPositions(arm_names[index], &candidate.arm[index]);
    for (std::size_t index = 0; index < hand_names.size(); ++index)
      state.setJointPositions(hand_names[index], &candidate.hand[index]);
    state.updateCollisionBodyTransforms();
    if (!state.satisfiesBounds(0.0))
      throw std::runtime_error("search generated a joint-limit violation");
    const Eigen::Isometry3d world_to_tcp = state.getGlobalLinkTransform(tcp_link);
    candidate.tcp_position_error =
        (world_to_tcp.translation() - Eigen::Vector3d(tcp_xy[0], tcp_xy[1], tcp_z)).norm();
    const double axis_dot = std::clamp(world_to_tcp.rotation().col(2).dot(-Eigen::Vector3d::UnitZ()), -1.0, 1.0);
    candidate.tcp_axis_error = std::acos(axis_dot);
    observed_maximum_tcp_position_error =
        std::max(observed_maximum_tcp_position_error, candidate.tcp_position_error);
    observed_maximum_tcp_axis_error = std::max(observed_maximum_tcp_axis_error, candidate.tcp_axis_error);
    if (candidate.tcp_position_error > maximum_tcp_position_error || candidate.tcp_axis_error > maximum_tcp_axis_error)
      throw std::runtime_error("interpolated IK family exceeded its FK error contract");
  };

  auto distanceSelf = [&](const collision_detection::AllowedCollisionMatrix& acm) {
    collision_detection::DistanceRequest request;
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::GLOBAL;
    request.acm = &acm;
    request.enableGroup(model);
    collision_detection::DistanceResult result;
    scene.getCollisionEnvUnpadded()->distanceSelf(request, result, state);
    return convertDistance(result);
  };

  auto isolatedRobotWorldDistance = [&](const std::vector<std::string>& target_ids,
                                        const std::vector<std::string>& target_links) {
    collision_detection::AllowedCollisionMatrix isolated = strict_acm;
    for (const std::string& link : model->getLinkModelNamesWithCollisionGeometry())
      for (const std::string& object_id : world_ids)
        isolated.setEntry(link, object_id, true);
    for (const std::string& link : target_links)
      for (const std::string& object_id : target_ids)
        isolated.setEntry(link, object_id, false);
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

  auto evaluate = [&](const double tcp_z, const double outer_blend, const double middle_blend) {
    Candidate candidate;
    candidate.evaluated = true;
    candidate.tcp_z = tcp_z;
    candidate.outer_blend = outer_blend;
    candidate.middle_blend = middle_blend;
    setState(tcp_z, outer_blend, middle_blend, candidate);
    ++evaluated_count;
    candidate.worst_nut_distance = -std::numeric_limits<double>::max();
    candidate.minimum_body_distance = std::numeric_limits<double>::max();
    candidate.maximum_nut_target_penetration = 0.0;
    for (const auto& group : finger_groups)
    {
      const DistanceSample nut_sample = isolatedRobotWorldDistance({ nut_id }, group.second);
      if (!pairContains(nut_sample, nut_id))
        throw std::runtime_error("isolated per-finger nut distance returned an unexpected pair");
      candidate.nut.emplace(group.first, nut_sample);
      candidate.worst_nut_distance = std::max(candidate.worst_nut_distance, nut_sample.distance);
      candidate.maximum_nut_target_penetration =
          std::max(candidate.maximum_nut_target_penetration, std::max(0.0, -nut_sample.distance));
    }
    ++finite_count;
    const double nut_margin = nut_touch_max - candidate.worst_nut_distance;
    candidate.minimum_margin = nut_margin;
    if (candidate.worst_nut_distance > nut_touch_max)
      return candidate;

    for (const auto& group : finger_groups)
    {
      const DistanceSample body_sample = isolatedRobotWorldDistance({ body_id }, group.second);
      if (!pairContains(body_sample, body_id))
        throw std::runtime_error("isolated per-finger body distance returned an unexpected pair");
      candidate.body.emplace(group.first, body_sample);
      candidate.minimum_body_distance = std::min(candidate.minimum_body_distance, body_sample.distance);
    }
    const double body_margin = candidate.minimum_body_distance - body_clearance_min;
    candidate.minimum_margin = std::min(nut_margin, body_margin);
    if (candidate.minimum_body_distance < body_clearance_min)
      return candidate;

    candidate.candidate_self = distanceSelf(candidate_acm);
    candidate.strict_self = distanceSelf(strict_acm);
    candidate.environment =
        isolatedRobotWorldDistance(environment_ids, model->getLinkModelNamesWithCollisionGeometry());
    std::vector<std::string> forbidden_endpoint_links;
    for (const std::string& link : model->getLinkModelNamesWithCollisionGeometry())
      if (all_finger_links.count(link) == 0U)
        forbidden_endpoint_links.push_back(link);
    candidate.forbidden_endpoint = isolatedRobotWorldDistance({ body_id, nut_id }, forbidden_endpoint_links);
    const bool all_distances_available = candidate.candidate_self.available && candidate.strict_self.available &&
                                         candidate.environment.available && candidate.forbidden_endpoint.available;
    if (!all_distances_available)
      throw std::runtime_error("FCL signed distance became unavailable during full nut-only gate");
    if (!pairContains(candidate.environment, table_id) && !pairContains(candidate.environment, fixture_id) &&
        !pairContains(candidate.environment, fixed_id))
      throw std::runtime_error("environment distance returned an unexpected object pair");
    if (!pairContains(candidate.forbidden_endpoint, body_id) && !pairContains(candidate.forbidden_endpoint, nut_id))
      throw std::runtime_error("forbidden endpoint distance returned an unexpected object pair");

    candidate.minimum_margin = std::min(
        { nut_margin, body_margin,
          candidate.candidate_self.distance - candidate_self_min,
          candidate.strict_self.distance - strict_self_min,
          candidate.environment.distance - environment_min,
          candidate.forbidden_endpoint.distance - body_clearance_min });
    candidate.feasible = candidate.worst_nut_distance <= nut_touch_max &&
                         candidate.minimum_body_distance >= body_clearance_min &&
                         candidate.candidate_self.distance >= candidate_self_min &&
                         candidate.strict_self.distance >= strict_self_min &&
                         candidate.environment.distance >= environment_min &&
                         candidate.forbidden_endpoint.distance >= body_clearance_min;
    return candidate;
  };

  Candidate best;
  Candidate best_feasible;
  auto consider = [&](const Candidate& candidate) {
    constexpr double epsilon = 1.0e-12;
    if (!best.evaluated || candidate.minimum_margin > best.minimum_margin + epsilon ||
        (std::abs(candidate.minimum_margin - best.minimum_margin) <= epsilon &&
         candidate.maximum_nut_target_penetration < best.maximum_nut_target_penetration))
      best = candidate;
    if (candidate.feasible &&
        (!best_feasible.evaluated || candidate.minimum_margin > best_feasible.minimum_margin + epsilon ||
         (std::abs(candidate.minimum_margin - best_feasible.minimum_margin) <= epsilon &&
          candidate.maximum_nut_target_penetration < best_feasible.maximum_nut_target_penetration)))
      best_feasible = candidate;
  };

  auto scanGrid = [&](const double z_lower, const double z_upper, const double z_step,
                      const double outer_lower, const double outer_upper, const double outer_step,
                      const double middle_lower, const double middle_upper, const double middle_step) {
    if (z_step <= 0.0 || outer_step <= 0.0 || middle_step <= 0.0 || z_lower > z_upper ||
        outer_lower > outer_upper || middle_lower > middle_upper)
      throw std::runtime_error("invalid search grid");
    const std::size_t z_count = static_cast<std::size_t>(std::floor((z_upper - z_lower) / z_step + 1.0e-9)) + 1U;
    const std::size_t outer_count =
        static_cast<std::size_t>(std::floor((outer_upper - outer_lower) / outer_step + 1.0e-9)) + 1U;
    const std::size_t middle_count =
        static_cast<std::size_t>(std::floor((middle_upper - middle_lower) / middle_step + 1.0e-9)) + 1U;
    for (std::size_t z_index = 0; z_index < z_count; ++z_index)
    {
      const double tcp_z = std::min(z_upper, z_lower + static_cast<double>(z_index) * z_step);
      for (std::size_t outer_index = 0; outer_index < outer_count; ++outer_index)
      {
        const double outer = std::min(outer_upper, outer_lower + static_cast<double>(outer_index) * outer_step);
        for (std::size_t middle_index = 0; middle_index < middle_count; ++middle_index)
        {
          const double middle =
              std::min(middle_upper, middle_lower + static_cast<double>(middle_index) * middle_step);
          consider(evaluate(tcp_z, outer, middle));
        }
      }
    }
  };

  const double global_z_lower = finiteValue(required(search, "tcp_z_lower_m"), "tcp_z_lower_m");
  const double global_z_upper = finiteValue(required(search, "tcp_z_upper_m"), "tcp_z_upper_m");
  const double coarse_z_step = finiteValue(required(search, "coarse_tcp_z_step_m"), "coarse_tcp_z_step_m");
  const double global_outer_lower =
      finiteValue(required(search, "coarse_outer_blend_lower"), "coarse_outer_blend_lower");
  const double global_outer_upper =
      finiteValue(required(search, "coarse_outer_blend_upper"), "coarse_outer_blend_upper");
  const double coarse_outer_step =
      finiteValue(required(search, "coarse_outer_blend_step"), "coarse_outer_blend_step");
  const double global_middle_lower =
      finiteValue(required(search, "coarse_middle_blend_lower"), "coarse_middle_blend_lower");
  const double global_middle_upper =
      finiteValue(required(search, "coarse_middle_blend_upper"), "coarse_middle_blend_upper");
  const double coarse_middle_step =
      finiteValue(required(search, "coarse_middle_blend_step"), "coarse_middle_blend_step");
  scanGrid(global_z_lower, global_z_upper, coarse_z_step, global_outer_lower, global_outer_upper,
           coarse_outer_step, global_middle_lower, global_middle_upper, coarse_middle_step);
  const std::size_t coarse_evaluated_count = evaluated_count;
  if (!best.evaluated)
    throw std::runtime_error("coarse search evaluated no candidates");

  const double refine_z_half =
      finiteValue(required(search, "refinement_tcp_half_width_m"), "refinement_tcp_half_width_m");
  const double refine_z_step =
      finiteValue(required(search, "refinement_tcp_z_step_m"), "refinement_tcp_z_step_m");
  const double refine_blend_half =
      finiteValue(required(search, "refinement_blend_half_width"), "refinement_blend_half_width");
  const double refine_blend_step =
      finiteValue(required(search, "refinement_blend_step"), "refinement_blend_step");
  const Candidate refinement_center = best_feasible.evaluated ? best_feasible : best;
  scanGrid(std::max(global_z_lower, refinement_center.tcp_z - refine_z_half),
           std::min(global_z_upper, refinement_center.tcp_z + refine_z_half), refine_z_step,
           std::max(global_outer_lower, refinement_center.outer_blend - refine_blend_half),
           std::min(global_outer_upper, refinement_center.outer_blend + refine_blend_half), refine_blend_step,
           std::max(global_middle_lower, refinement_center.middle_blend - refine_blend_half),
           std::min(global_middle_upper, refinement_center.middle_blend + refine_blend_half), refine_blend_step);

  const bool nut_only_command_candidate_found = best_feasible.evaluated;
  PathAudit path_audit;
  if (nut_only_command_candidate_found)
  {
    path_audit.evaluated = true;
    const YAML::Node path = required(config, "discrete_regrasp_path");
    const double rate_hz = finiteValue(required(path, "rate_hz"), "path.rate_hz");
    if (rate_hz != 240.0 || required(path, "interpolation").as<std::string>() != "minimum_jerk")
      throw std::runtime_error("discrete regrasp path must use 240 Hz minimum jerk");
    const double carry_tcp_z = finiteValue(required(path, "carry_tcp_z_m"), "carry_tcp_z_m");
    const std::array<double, 7> carry_arm = armAt(carry_tcp_z);
    const std::vector<double> raw_carry_hand = vectorOf(required(path, "carry_hand_rad"), 4U, "carry_hand_rad");
    const std::vector<double> raw_open_hand = vectorOf(required(path, "open_hand_rad"), 4U, "open_hand_rad");
    std::array<double, 4> carry_hand{};
    std::array<double, 4> open_hand{};
    std::copy(raw_carry_hand.begin(), raw_carry_hand.end(), carry_hand.begin());
    std::copy(raw_open_hand.begin(), raw_open_hand.end(), open_hand.begin());
    const double release_duration = finiteValue(required(path, "release_duration_s"), "release_duration_s");
    const double reposition_duration =
        finiteValue(required(path, "open_reposition_duration_s"), "open_reposition_duration_s");
    const double closure_duration =
        finiteValue(required(path, "nut_closure_duration_s"), "nut_closure_duration_s");
    const auto exactSteps = [rate_hz](const double duration, const std::string& name) {
      const double raw = duration * rate_hz;
      const auto result = static_cast<std::size_t>(std::llround(raw));
      if (duration <= 0.0 || std::abs(raw - static_cast<double>(result)) > 1.0e-9)
        throw std::runtime_error(name + " must contain a positive whole number of 240 Hz steps");
      return result;
    };
    const std::size_t release_steps = exactSteps(release_duration, "release_duration_s");
    const std::size_t reposition_steps = exactSteps(reposition_duration, "open_reposition_duration_s");
    const std::size_t closure_steps = exactSteps(closure_duration, "nut_closure_duration_s");
    path_audit.expected_sample_count =
        required(path, "expected_sample_count").as<std::size_t>();
    if (release_steps + reposition_steps + closure_steps != path_audit.expected_sample_count)
      throw std::runtime_error("discrete regrasp path sample count mismatch");
    const double maximum_joint_step =
        finiteValue(required(path, "maximum_joint_step_rad"), "maximum_joint_step_rad");
    const double open_endpoint_clearance = finiteValue(
        required(path, "open_reposition_endpoint_clearance_min_m"),
        "open_reposition_endpoint_clearance_min_m");
    if (maximum_joint_step != 0.0025 || open_endpoint_clearance != 0.001)
      throw std::runtime_error("discrete regrasp path thresholds are not canonical v1");

    std::vector<std::string> forbidden_endpoint_links;
    for (const std::string& link : model->getLinkModelNamesWithCollisionGeometry())
      if (all_finger_links.count(link) == 0U)
        forbidden_endpoint_links.push_back(link);
    std::array<double, 11> previous_command{};
    for (std::size_t index = 0; index < carry_arm.size(); ++index)
      previous_command[index] = carry_arm[index];
    for (std::size_t index = 0; index < carry_hand.size(); ++index)
      previous_command[7U + index] = carry_hand[index];

    auto inspectPathSample = [&](const std::string& phase, const std::size_t phase_step,
                                 const std::array<double, 7>& arm, const std::array<double, 4>& hand,
                                 const bool require_open_endpoint_clearance,
                                 const bool require_body_clearance) {
      for (std::size_t index = 0; index < arm_names.size(); ++index)
        state.setJointPositions(arm_names[index], &arm[index]);
      for (std::size_t index = 0; index < hand_names.size(); ++index)
        state.setJointPositions(hand_names[index], &hand[index]);
      state.updateCollisionBodyTransforms();
      if (!state.satisfiesBounds(0.0))
        ++path_audit.bounds_violation_count;
      std::array<double, 11> command{};
      for (std::size_t index = 0; index < arm.size(); ++index)
        command[index] = arm[index];
      for (std::size_t index = 0; index < hand.size(); ++index)
        command[7U + index] = hand[index];
      for (std::size_t index = 0; index < command.size(); ++index)
        path_audit.maximum_joint_step =
            std::max(path_audit.maximum_joint_step, std::abs(command[index] - previous_command[index]));
      previous_command = command;
      ++path_audit.sample_count;

      updateMinimum(path_audit.candidate_self, distanceSelf(candidate_acm), phase, phase_step,
                    path_audit.sample_count);
      updateMinimum(path_audit.strict_self, distanceSelf(strict_acm), phase, phase_step, path_audit.sample_count);
      updateMinimum(path_audit.environment,
                    isolatedRobotWorldDistance(environment_ids, model->getLinkModelNamesWithCollisionGeometry()),
                    phase, phase_step, path_audit.sample_count);
      updateMinimum(path_audit.nonfinger_endpoint,
                    isolatedRobotWorldDistance({ body_id, nut_id }, forbidden_endpoint_links), phase,
                    phase_step, path_audit.sample_count);
      if (require_body_clearance || require_open_endpoint_clearance)
      {
        for (const auto& group : finger_groups)
        {
          if (require_body_clearance)
            updateMinimum(path_audit.body_during_reapproach,
                          isolatedRobotWorldDistance({ body_id }, group.second), phase, phase_step,
                          path_audit.sample_count);
          if (require_open_endpoint_clearance)
            updateMinimum(path_audit.nut_during_open_reposition,
                          isolatedRobotWorldDistance({ nut_id }, group.second), phase, phase_step,
                          path_audit.sample_count);
        }
      }
    };

    auto interpolate = [](const auto& start, const auto& target, const double blend) {
      auto result = start;
      for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = start[index] + blend * (target[index] - start[index]);
      return result;
    };
    for (std::size_t index = 0; index < release_steps; ++index)
    {
      const double fraction = static_cast<double>(index + 1U) / static_cast<double>(release_steps);
      inspectPathSample("release_mixed_grip", index + 1U, carry_arm,
                        interpolate(carry_hand, open_hand, minimumJerk(fraction)), false, false);
    }
    for (std::size_t index = 0; index < reposition_steps; ++index)
    {
      const double fraction = static_cast<double>(index + 1U) / static_cast<double>(reposition_steps);
      inspectPathSample("open_reposition_to_nut_band", index + 1U,
                        interpolate(carry_arm, best_feasible.arm, minimumJerk(fraction)), open_hand, true, true);
    }
    for (std::size_t index = 0; index < closure_steps; ++index)
    {
      const double fraction = static_cast<double>(index + 1U) / static_cast<double>(closure_steps);
      inspectPathSample("nut_only_closure", index + 1U, best_feasible.arm,
                        interpolate(open_hand, best_feasible.hand, minimumJerk(fraction)), false, true);
    }

    path_audit.passed = path_audit.sample_count == path_audit.expected_sample_count &&
                        path_audit.bounds_violation_count == 0U &&
                        path_audit.maximum_joint_step <= maximum_joint_step &&
                        path_audit.candidate_self.available &&
                        path_audit.candidate_self.sample.distance >= candidate_self_min &&
                        path_audit.strict_self.available &&
                        path_audit.strict_self.sample.distance >= strict_self_min &&
                        path_audit.environment.available &&
                        path_audit.environment.sample.distance >= environment_min &&
                        path_audit.nonfinger_endpoint.available &&
                        path_audit.nonfinger_endpoint.sample.distance >= body_clearance_min &&
                        path_audit.body_during_reapproach.available &&
                        path_audit.body_during_reapproach.sample.distance >= body_clearance_min &&
                        path_audit.nut_during_open_reposition.available &&
                        path_audit.nut_during_open_reposition.sample.distance >= open_endpoint_clearance;
  }
  const bool continuous_collision_verified = false;
  const auto finished = std::chrono::steady_clock::now();
  const double wall_seconds = std::chrono::duration<double>(finished - started).count();

  YAML::Node report;
  report["schema_version"] = kReportSchema;
  if (!nut_only_command_candidate_found)
    report["status"] = "FAIL_CLOSED_NO_NUT_ONLY_COMMAND_CANDIDATE";
  else if (!path_audit.passed)
    report["status"] = "FAIL_CLOSED_DISCRETE_REGRASP_PATH";
  else if (!continuous_collision_verified)
    report["status"] = "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED";
  else
    report["status"] = "PASSED";
  report["nut_only_command_candidate_found"] = nut_only_command_candidate_found;
  report["candidate_may_proceed_to_physx_static_ab"] =
      nut_only_command_candidate_found && path_audit.passed;
  report["continuous_collision_verified"] = continuous_collision_verified;
  report["physics_contact_verified"] = false;
  report["force_closure_verified"] = false;
  report["assembly_success_claimed"] = false;
  report["project_root"] = arguments.project_root.string();
  report["config_path"] = arguments.config.string();
  report["config_sha256"] = sha256File(arguments.config);
  for (const auto& entry : inputs)
  {
    report["inputs"][entry.first]["path"] = entry.second.relative_path.string();
    report["inputs"][entry.first]["expected_sha256"] = entry.second.expected_hash;
    report["inputs"][entry.first]["actual_sha256"] = entry.second.actual_hash;
    report["inputs"][entry.first]["hash_matches"] = entry.second.expected_hash == entry.second.actual_hash;
  }
  report["engage_geometry"]["fixed_origin_world_m"].push_back(fixed_origin.x());
  report["engage_geometry"]["fixed_origin_world_m"].push_back(fixed_origin.y());
  report["engage_geometry"]["fixed_origin_world_m"].push_back(fixed_origin.z());
  report["engage_geometry"]["engage_gap_m"] = engage_gap;
  report["engage_geometry"]["body_root_world_m"].push_back(body_root.x());
  report["engage_geometry"]["body_root_world_m"].push_back(body_root.y());
  report["engage_geometry"]["body_root_world_m"].push_back(body_root.z());
  report["engage_geometry"]["body_world_id"] = body_id;
  report["engage_geometry"]["nut_world_id"] = nut_id;
  report["method"]["collision_detector"] = scene.getCollisionDetectorName();
  report["method"]["candidate_srdf_never_pairs_reenabled_for_strict"] = restored_never_count;
  report["method"]["coarse_candidate_count"] = coarse_evaluated_count;
  report["method"]["total_candidate_count"] = evaluated_count;
  report["method"]["finite_candidate_count"] = finite_count;
  report["method"]["wall_seconds"] = wall_seconds;
  report["method"]["observed_maximum_interpolated_tcp_position_error_m"] =
      observed_maximum_tcp_position_error;
  report["method"]["observed_maximum_interpolated_tcp_axis_error_rad"] = observed_maximum_tcp_axis_error;
  report["acceptance"]["nut_touch_signed_distance_max_m"] = nut_touch_max;
  report["acceptance"]["body_signed_clearance_min_m"] = body_clearance_min;
  report["acceptance"]["candidate_self_clearance_min_m"] = candidate_self_min;
  report["acceptance"]["strict_self_clearance_min_m"] = strict_self_min;
  report["acceptance"]["robot_environment_clearance_min_m"] = environment_min;
  report["best_overall"] = candidateNode(best);
  report["best_feasible"] = candidateNode(best_feasible);
  report["discrete_regrasp_path"]["evaluated"] = path_audit.evaluated;
  report["discrete_regrasp_path"]["passed"] = path_audit.passed;
  report["discrete_regrasp_path"]["sample_count"] = path_audit.sample_count;
  report["discrete_regrasp_path"]["expected_sample_count"] = path_audit.expected_sample_count;
  report["discrete_regrasp_path"]["bounds_violation_count"] = path_audit.bounds_violation_count;
  report["discrete_regrasp_path"]["maximum_joint_step_rad"] = path_audit.maximum_joint_step;
  report["discrete_regrasp_path"]["candidate_self_minimum"] = pathMinimumNode(path_audit.candidate_self);
  report["discrete_regrasp_path"]["strict_self_minimum"] = pathMinimumNode(path_audit.strict_self);
  report["discrete_regrasp_path"]["environment_minimum"] = pathMinimumNode(path_audit.environment);
  report["discrete_regrasp_path"]["nonfinger_endpoint_minimum"] =
      pathMinimumNode(path_audit.nonfinger_endpoint);
  report["discrete_regrasp_path"]["body_during_reapproach_minimum"] =
      pathMinimumNode(path_audit.body_during_reapproach);
  report["discrete_regrasp_path"]["nut_during_open_reposition_minimum"] =
      pathMinimumNode(path_audit.nut_during_open_reposition);
  report["interpretation"] =
      "A feasible command means all three finger groups reach the separated nut proxy while every finger remains at "
      "least 1 mm from BodyAssembly, strict/candidate self clearance remains at least 1 mm, non-finger robot links "
      "remain at least 1 mm from both endpoint components, and table/fixture/receptacle clearance remains at least "
      "5 mm. Negative nut distance is only a torque-limited preload command, never a rigid equilibrium proof.";
  report["limitations"]["continuous_collision_reason"] =
      "Humble FCL does not provide the required continuous robot/world and self-collision proof for this trajectory.";
  report["limitations"]["separated_body_nut_are_static_world_proxies"] = true;
  report["limitations"]["revolute_joint_dynamics_modeled"] = false;
  report["limitations"]["finger_torque_limit_nm"] = 2.0;
  report["limitations"]["operational_finger_target_nm"] = 1.8;
  report["limitations"]["thread_teeth_or_real_pitch_modeled"] = false;
  writeReport(report, arguments);

  if (arguments.report_only)
    return 0;
  return nut_only_command_candidate_found && continuous_collision_verified ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    return runSearch(parseArguments(argc, argv));
  }
  catch (const std::exception& error)
  {
    YAML::Node report;
    report["schema_version"] = kReportSchema;
    report["status"] = "FAIL_CLOSED_RUNTIME_ERROR";
    report["nut_only_command_candidate_found"] = false;
    report["error"] = error.what();
    YAML::Emitter emitter;
    emitter << report;
    std::cerr << emitter.c_str() << '\n';
    return 1;
  }
}
