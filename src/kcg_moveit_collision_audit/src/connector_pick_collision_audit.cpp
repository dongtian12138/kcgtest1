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
#include <cstdint>
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
constexpr char kSchemaVersion[] = "kcg_moveit_connector_pick_collision_audit_v1";
constexpr char kReportSchemaVersion[] = "kcg_moveit_connector_pick_collision_audit_report_v1";

using JointValues = std::vector<double>;

struct Arguments
{
  fs::path project_root;
  fs::path config;
  fs::path output;
  bool has_output{ false };
  bool report_only{ false };
};

struct InputRecord
{
  std::string name;
  fs::path relative_path;
  fs::path absolute_path;
  std::string expected_sha256;
  std::string actual_sha256;
};

struct MinimumRecord
{
  bool available{ false };
  double distance{ std::numeric_limits<double>::max() };
  std::string phase;
  std::size_t sample_index{ 0 };
  std::size_t phase_sample_index{ 0 };
  std::array<std::string, 2> links;
  std::array<std::string, 2> body_types;
  std::array<Eigen::Vector3d, 2> nearest_points{ Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero() };
};

struct CollisionExample
{
  std::string category;
  std::string phase;
  std::size_t sample_index{ 0 };
  std::size_t phase_sample_index{ 0 };
  std::string link1;
  std::string link2;
};

struct PolicyStats
{
  std::string name;
  std::size_t self_collision_samples{ 0 };
  std::size_t environment_collision_samples{ 0 };
  std::size_t robot_link_environment_collision_samples{ 0 };
  std::size_t attached_environment_collision_samples{ 0 };
  std::size_t robot_link_table_collision_samples{ 0 };
  std::size_t self_clearance_violations{ 0 };
  std::size_t environment_clearance_violations{ 0 };
  MinimumRecord minimum_self_distance;
  MinimumRecord minimum_environment_distance;
  MinimumRecord minimum_robot_link_table_distance;
  bool has_first_self_collision{ false };
  CollisionExample first_self_collision;
  bool has_first_robot_link_table_collision{ false };
  CollisionExample first_robot_link_table_collision;
  std::vector<CollisionExample> examples;
};

struct ProxyGeometry
{
  std::vector<shapes::ShapeConstPtr> loose_shapes;
  EigenSTL::vector_Isometry3d loose_shape_poses;
  Eigen::Isometry3d loose_world_pose{ Eigen::Isometry3d::Identity() };
};

enum class SceneMode
{
  PRECLOSURE,
  INTENTIONAL_FINGER_CONTACT,
  ATTACHED
};

std::string usage()
{
  return "usage: connector_pick_collision_audit --project-root PATH --config PATH "
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
  if (!fs::is_directory(result.project_root))
    throw std::runtime_error("project root is not a directory: " + result.project_root.string());
  if (!result.config.is_absolute())
    result.config = result.project_root / result.config;
  result.config = fs::canonical(result.config);
  if (result.has_output && !result.output.is_absolute())
    result.output = result.project_root / result.output;
  return result;
}

bool isWithin(const fs::path& candidate, const fs::path& root)
{
  const fs::path relative = candidate.lexically_relative(root);
  return !relative.empty() && *relative.begin() != "..";
}

fs::path resolveProjectInput(const fs::path& root, const std::string& relative_text)
{
  const fs::path relative(relative_text);
  if (relative.empty() || relative.is_absolute())
    throw std::runtime_error("input paths must be nonempty and relative to project root: " + relative_text);
  const fs::path resolved = fs::canonical(root / relative);
  if (!isWithin(resolved, root) || !fs::is_regular_file(resolved))
    throw std::runtime_error("input path escapes project root or is not a file: " + relative_text);
  return resolved;
}

std::string sha256File(const fs::path& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open for SHA-256: " + path.string());

  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("failed to initialize SHA-256");

  std::array<char, 65536> buffer{};
  while (stream)
  {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = stream.gcount();
    if (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(count)) != 1)
      throw std::runtime_error("failed while computing SHA-256 for " + path.string());
  }
  if (!stream.eof())
    throw std::runtime_error("failed while reading for SHA-256: " + path.string());

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 || digest_size != 32U)
    throw std::runtime_error("failed to finalize SHA-256 for " + path.string());

  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index)
    encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
  return encoded.str();
}

std::string readTextFile(const fs::path& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open text input: " + path.string());
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof())
    throw std::runtime_error("failed to read text input: " + path.string());
  return buffer.str();
}

YAML::Node requireNode(const YAML::Node& parent, const std::string& key)
{
  const YAML::Node value = parent[key];
  if (!value)
    throw std::runtime_error("missing required YAML key: " + key);
  return value;
}

double finiteDouble(const YAML::Node& node, const std::string& name)
{
  const double value = node.as<double>();
  if (!std::isfinite(value))
    throw std::runtime_error(name + " must be finite");
  return value;
}

JointValues vectorOf(const YAML::Node& node, const std::size_t expected_size, const std::string& name)
{
  if (!node.IsSequence() || node.size() != expected_size)
    throw std::runtime_error(name + " must contain exactly " + std::to_string(expected_size) + " values");
  JointValues result;
  result.reserve(expected_size);
  for (std::size_t index = 0; index < expected_size; ++index)
    result.push_back(finiteDouble(node[index], name + "[" + std::to_string(index) + "]"));
  return result;
}

Eigen::Vector3d vector3Of(const YAML::Node& node, const std::string& name)
{
  const JointValues values = vectorOf(node, 3, name);
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

Eigen::Isometry3d translated(const Eigen::Vector3d& value)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = value;
  return pose;
}

double minimumJerkBlend(const double fraction)
{
  if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
    throw std::runtime_error("minimum-jerk fraction must be finite and in [0, 1]");
  return fraction * fraction * fraction * (10.0 + fraction * (-15.0 + 6.0 * fraction));
}

JointValues interpolate(const JointValues& start, const JointValues& target, const double blend)
{
  if (start.size() != target.size())
    throw std::runtime_error("joint interpolation vector sizes do not match");
  JointValues result(start.size(), 0.0);
  for (std::size_t index = 0; index < start.size(); ++index)
    result[index] = start[index] + blend * (target[index] - start[index]);
  return result;
}

std::size_t exactSteps(const double duration, const double rate, const std::string& name)
{
  if (!std::isfinite(duration) || duration <= 0.0 || !std::isfinite(rate) || rate <= 0.0)
    throw std::runtime_error(name + " duration and sample rate must be positive and finite");
  const double raw = duration * rate;
  const auto rounded = static_cast<std::size_t>(std::llround(raw));
  if (std::abs(raw - static_cast<double>(rounded)) > 1.0e-9)
    throw std::runtime_error(name + " does not contain a whole number of samples");
  return rounded;
}

std::string bodyTypeName(const collision_detection::BodyType type)
{
  switch (type)
  {
    case collision_detection::BodyType::ROBOT_LINK:
      return "ROBOT_LINK";
    case collision_detection::BodyType::ROBOT_ATTACHED:
      return "ROBOT_ATTACHED";
    case collision_detection::BodyType::WORLD_OBJECT:
      return "WORLD_OBJECT";
  }
  return "UNKNOWN";
}

void updateMinimum(MinimumRecord& record, const collision_detection::DistanceResult& result,
                   const std::string& phase, const std::size_t sample_index,
                   const std::size_t phase_sample_index)
{
  const auto& candidate = result.minimum_distance;
  if (!std::isfinite(candidate.distance) || candidate.distance >= record.distance)
    return;
  record.available = true;
  record.distance = candidate.distance;
  record.phase = phase;
  record.sample_index = sample_index;
  record.phase_sample_index = phase_sample_index;
  for (std::size_t index = 0; index < 2; ++index)
  {
    record.links[index] = candidate.link_names[index];
    record.body_types[index] = bodyTypeName(candidate.body_types[index]);
    record.nearest_points[index] = candidate.nearest_points[index];
  }
}

void recordCollisionExamples(PolicyStats& stats, const collision_detection::CollisionResult& result,
                             const std::string& category, const std::string& phase, const std::size_t sample_index,
                             const std::size_t phase_sample_index, const std::size_t maximum_examples)
{
  if (stats.examples.size() >= maximum_examples)
    return;
  if (result.contacts.empty())
  {
    stats.examples.push_back({ category, phase, sample_index, phase_sample_index, "UNAVAILABLE", "UNAVAILABLE" });
    return;
  }
  for (const auto& entry : result.contacts)
  {
    if (stats.examples.size() >= maximum_examples)
      return;
    stats.examples.push_back(
        { category, phase, sample_index, phase_sample_index, entry.first.first, entry.first.second });
  }
}

CollisionExample firstCollisionExample(const collision_detection::CollisionResult& result, const std::string& category,
                                       const std::string& phase, const std::size_t sample_index,
                                       const std::size_t phase_sample_index)
{
  if (result.contacts.empty())
    return { category, phase, sample_index, phase_sample_index, "UNAVAILABLE", "UNAVAILABLE" };
  const auto& first = *result.contacts.begin();
  return { category, phase, sample_index, phase_sample_index, first.first.first, first.first.second };
}

YAML::Node collisionExampleNode(const CollisionExample& example)
{
  YAML::Node item;
  item["category"] = example.category;
  item["phase"] = example.phase;
  item["sample_index"] = example.sample_index;
  item["sample_index_zero_based"] = example.sample_index;
  item["phase_sample_index_zero_based"] = example.phase_sample_index;
  item["phase_step_one_based"] = example.phase_sample_index + 1U;
  item["pair"].push_back(example.link1);
  item["pair"].push_back(example.link2);
  return item;
}

void inspectPolicy(const planning_scene::PlanningScene& scene, const moveit::core::RobotState& state,
                   const collision_detection::AllowedCollisionMatrix& acm, const std::string& phase,
                   const std::size_t sample_index, const std::size_t phase_sample_index, const double minimum_clearance,
                   const std::size_t maximum_examples,
                   const collision_detection::AllowedCollisionMatrix& robot_table_acm,
                   PolicyStats& stats)
{
  collision_detection::CollisionRequest collision_request;
  collision_request.contacts = true;
  collision_request.max_contacts = 128;
  collision_request.max_contacts_per_pair = 8;

  collision_detection::CollisionResult self_collision;
  scene.getCollisionEnvUnpadded()->checkSelfCollision(collision_request, self_collision, state, acm);
  if (self_collision.collision)
  {
    ++stats.self_collision_samples;
    if (!stats.has_first_self_collision)
    {
      stats.has_first_self_collision = true;
      stats.first_self_collision =
          firstCollisionExample(self_collision, "self", phase, sample_index, phase_sample_index);
    }
    recordCollisionExamples(stats, self_collision, "self", phase, sample_index, phase_sample_index, maximum_examples);
  }

  collision_detection::CollisionResult environment_collision;
  scene.getCollisionEnv()->checkRobotCollision(collision_request, environment_collision, state, acm);
  if (environment_collision.collision)
  {
    ++stats.environment_collision_samples;
    recordCollisionExamples(stats, environment_collision, "environment", phase, sample_index, phase_sample_index,
                            maximum_examples);
    bool has_robot_link = false;
    bool has_attached = false;
    for (const auto& entry : environment_collision.contacts)
      for (const collision_detection::Contact& contact : entry.second)
      {
        has_robot_link = has_robot_link || contact.body_type_1 == collision_detection::BodyType::ROBOT_LINK ||
                         contact.body_type_2 == collision_detection::BodyType::ROBOT_LINK;
        has_attached = has_attached || contact.body_type_1 == collision_detection::BodyType::ROBOT_ATTACHED ||
                       contact.body_type_2 == collision_detection::BodyType::ROBOT_ATTACHED;
      }
    if (has_robot_link)
      ++stats.robot_link_environment_collision_samples;
    if (has_attached)
      ++stats.attached_environment_collision_samples;
  }

  collision_detection::CollisionResult robot_table_collision;
  scene.getCollisionEnv()->checkRobotCollision(collision_request, robot_table_collision, state, robot_table_acm);
  if (robot_table_collision.collision)
  {
    ++stats.robot_link_table_collision_samples;
    if (!stats.has_first_robot_link_table_collision)
    {
      stats.has_first_robot_link_table_collision = true;
      stats.first_robot_link_table_collision =
          firstCollisionExample(robot_table_collision, "robot_link_table", phase, sample_index, phase_sample_index);
    }
    recordCollisionExamples(stats, robot_table_collision, "robot_link_table", phase, sample_index, phase_sample_index,
                            maximum_examples);
  }

  collision_detection::DistanceRequest distance_request;
  distance_request.enable_nearest_points = true;
  distance_request.enable_signed_distance = true;
  distance_request.type = collision_detection::DistanceRequestType::GLOBAL;
  distance_request.acm = &acm;
  distance_request.enableGroup(scene.getRobotModel());

  collision_detection::DistanceResult self_distance;
  scene.getCollisionEnvUnpadded()->distanceSelf(distance_request, self_distance, state);
  updateMinimum(stats.minimum_self_distance, self_distance, phase, sample_index, phase_sample_index);
  if (std::isfinite(self_distance.minimum_distance.distance) &&
      self_distance.minimum_distance.distance < minimum_clearance)
    ++stats.self_clearance_violations;

  collision_detection::DistanceResult environment_distance;
  scene.getCollisionEnv()->distanceRobot(distance_request, environment_distance, state);
  updateMinimum(stats.minimum_environment_distance, environment_distance, phase, sample_index, phase_sample_index);
  if (std::isfinite(environment_distance.minimum_distance.distance) &&
      environment_distance.minimum_distance.distance < minimum_clearance)
    ++stats.environment_clearance_violations;

  distance_request.acm = &robot_table_acm;
  collision_detection::DistanceResult robot_table_distance;
  scene.getCollisionEnv()->distanceRobot(distance_request, robot_table_distance, state);
  updateMinimum(stats.minimum_robot_link_table_distance, robot_table_distance, phase, sample_index, phase_sample_index);
}

YAML::Node minimumNode(const MinimumRecord& record)
{
  YAML::Node node;
  node["available"] = record.available;
  if (!record.available)
    return node;
  node["distance_m"] = record.distance;
  node["phase"] = record.phase;
  node["sample_index"] = record.sample_index;
  node["sample_index_zero_based"] = record.sample_index;
  node["phase_sample_index_zero_based"] = record.phase_sample_index;
  node["phase_step_one_based"] = record.phase_sample_index + 1U;
  node["pair"].push_back(record.links[0]);
  node["pair"].push_back(record.links[1]);
  node["body_types"].push_back(record.body_types[0]);
  node["body_types"].push_back(record.body_types[1]);
  for (const Eigen::Vector3d& point : record.nearest_points)
  {
    YAML::Node point_node;
    point_node.push_back(point.x());
    point_node.push_back(point.y());
    point_node.push_back(point.z());
    node["nearest_points_world_m"].push_back(point_node);
  }
  return node;
}

YAML::Node policyNode(const PolicyStats& stats)
{
  YAML::Node node;
  node["self_collision_samples"] = stats.self_collision_samples;
  node["environment_collision_samples"] = stats.environment_collision_samples;
  node["robot_link_environment_collision_samples"] = stats.robot_link_environment_collision_samples;
  node["attached_environment_collision_samples"] = stats.attached_environment_collision_samples;
  node["robot_link_table_collision_samples"] = stats.robot_link_table_collision_samples;
  node["self_clearance_violations"] = stats.self_clearance_violations;
  node["environment_clearance_violations"] = stats.environment_clearance_violations;
  node["minimum_self_distance"] = minimumNode(stats.minimum_self_distance);
  node["minimum_environment_distance"] = minimumNode(stats.minimum_environment_distance);
  node["minimum_robot_link_table_distance"] = minimumNode(stats.minimum_robot_link_table_distance);
  node["first_self_collision"]["available"] = stats.has_first_self_collision;
  if (stats.has_first_self_collision)
    node["first_self_collision"]["record"] = collisionExampleNode(stats.first_self_collision);
  node["first_robot_link_table_collision"]["available"] = stats.has_first_robot_link_table_collision;
  if (stats.has_first_robot_link_table_collision)
    node["first_robot_link_table_collision"]["record"] =
        collisionExampleNode(stats.first_robot_link_table_collision);
  for (const CollisionExample& example : stats.examples)
    node["collision_examples"].push_back(collisionExampleNode(example));
  if (stats.examples.empty())
    node["collision_examples"] = YAML::Node(YAML::NodeType::Sequence);
  return node;
}

void writeReport(const YAML::Node& report, const Arguments& arguments)
{
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << report;
  if (!emitter.good())
    throw std::runtime_error("failed to serialize audit report");
  const std::string text = std::string(emitter.c_str()) + "\n";
  std::cout << text;
  if (arguments.has_output)
  {
    fs::create_directories(arguments.output.parent_path());
    std::ofstream stream(arguments.output);
    if (!stream)
      throw std::runtime_error("cannot open report output: " + arguments.output.string());
    stream << text;
    if (!stream)
      throw std::runtime_error("failed to write report output: " + arguments.output.string());
  }
}

int runAudit(const Arguments& arguments)
{
  const YAML::Node audit_config = YAML::LoadFile(arguments.config.string());
  if (requireNode(audit_config, "schema_version").as<std::string>() != kSchemaVersion)
    throw std::runtime_error("unsupported audit config schema");
  const std::string profile = requireNode(audit_config, "profile").as<std::string>();
  const bool is_d38999 = profile == "d38999_shell25j_v1";
  if (!is_d38999 && profile != "synthetic_connector_v1")
    throw std::runtime_error("unsupported audit profile: " + profile);

  const YAML::Node inputs_node = requireNode(audit_config, "inputs");
  const std::vector<std::string> required_inputs =
      is_d38999 ? std::vector<std::string>{ "urdf_xacro", "candidate_srdf", "d38999_pick", "d38999_scene",
                                          "d38999_proxy" }
                  : std::vector<std::string>{ "urdf_xacro", "candidate_srdf", "home_to_pregrasp", "tabletop_pick",
                                              "tabletop_scene", "connector_task" };
  if (!inputs_node.IsMap() || inputs_node.size() != required_inputs.size())
    throw std::runtime_error("audit input set does not exactly match the selected profile");
  std::map<std::string, InputRecord> inputs;
  for (const std::string& name : required_inputs)
  {
    const YAML::Node item = requireNode(inputs_node, name);
    InputRecord record;
    record.name = name;
    record.relative_path = requireNode(item, "path").as<std::string>();
    record.expected_sha256 = requireNode(item, "sha256").as<std::string>();
    record.absolute_path = resolveProjectInput(arguments.project_root, record.relative_path.string());
    record.actual_sha256 = sha256File(record.absolute_path);
    if (record.expected_sha256.size() != 64U || record.actual_sha256 != record.expected_sha256)
      throw std::runtime_error("content hash mismatch for input " + name + ": expected=" + record.expected_sha256 +
                               " actual=" + record.actual_sha256);
    inputs.emplace(name, std::move(record));
  }

  YAML::Node robot_yaml;
  YAML::Node home_motion_yaml;
  YAML::Node pick_motion_yaml;
  YAML::Node approach_segments_yaml;
  YAML::Node tabletop_yaml;
  YAML::Node geometry_yaml;
  double pregrasp_hold_duration = 0.0;
  if (is_d38999)
  {
    const YAML::Node pick_yaml = YAML::LoadFile(inputs.at("d38999_pick").absolute_path.string());
    tabletop_yaml = YAML::LoadFile(inputs.at("d38999_scene").absolute_path.string());
    const YAML::Node proxy_yaml = YAML::LoadFile(inputs.at("d38999_proxy").absolute_path.string());
    if (requireNode(pick_yaml, "schema_version").as<std::string>() != "kcg_d38999_tabletop_pick_v1" ||
        requireNode(tabletop_yaml, "schema_version").as<std::string>() != "kcg_d38999_tabletop_scene_v1" ||
        requireNode(proxy_yaml, "schema_version").as<std::string>() != "kcg_d38999_shell25j_proxy_v1")
      throw std::runtime_error("one or more D38999 trajectory/scene/proxy schemas are unsupported");
    robot_yaml = pick_yaml["robot"];
    home_motion_yaml = pick_yaml["motion"];
    pick_motion_yaml = pick_yaml["motion"];
    approach_segments_yaml = pick_yaml["motion"]["approach_segments"];
    geometry_yaml = proxy_yaml["proxy_geometry_m"];
    pregrasp_hold_duration =
        finiteDouble(pick_yaml["motion"]["pregrasp_hold_duration_s"], "pregrasp_hold_duration_s");
  }
  else
  {
    const YAML::Node home_yaml = YAML::LoadFile(inputs.at("home_to_pregrasp").absolute_path.string());
    const YAML::Node pick_yaml = YAML::LoadFile(inputs.at("tabletop_pick").absolute_path.string());
    tabletop_yaml = YAML::LoadFile(inputs.at("tabletop_scene").absolute_path.string());
    const YAML::Node task_yaml = YAML::LoadFile(inputs.at("connector_task").absolute_path.string());
    if (requireNode(home_yaml, "schema_version").as<std::string>() != "kcg_connector_home_to_pregrasp_v1" ||
        requireNode(pick_yaml, "schema_version").as<std::string>() != "kcg_connector_tabletop_pick_v1" ||
        requireNode(tabletop_yaml, "schema_version").as<std::string>() != "kcg_connector_tabletop_scene_v1" ||
        requireNode(task_yaml, "task_version").as<std::string>() != "kcg_connector_task_v1")
      throw std::runtime_error("one or more synthetic trajectory/scene/task schemas are unsupported");
    robot_yaml = home_yaml["robot"];
    home_motion_yaml = home_yaml["motion"];
    pick_motion_yaml = pick_yaml["motion"];
    approach_segments_yaml = home_yaml["motion"]["segments"];
    geometry_yaml = task_yaml["geometry"];
    pregrasp_hold_duration = finiteDouble(home_yaml["motion"]["hold_duration_s"], "hold_duration_s");
  }

  const YAML::Node sampling = requireNode(audit_config, "sampling");
  const double rate_hz = finiteDouble(requireNode(sampling, "rate_hz"), "sampling.rate_hz");
  if (requireNode(sampling, "interpolation").as<std::string>() != "minimum_jerk" ||
      requireNode(home_motion_yaml, "interpolation").as<std::string>() != "minimum_jerk" ||
      requireNode(pick_motion_yaml, "interpolation").as<std::string>() != "minimum_jerk")
    throw std::runtime_error("only minimum_jerk interpolation is accepted");
  const std::size_t expected_sample_count = requireNode(sampling, "expected_sample_count").as<std::size_t>();
  const double maximum_joint_step = finiteDouble(requireNode(sampling, "maximum_joint_step_rad"),
                                                 "sampling.maximum_joint_step_rad");
  if (maximum_joint_step <= 0.0)
    throw std::runtime_error("sampling.maximum_joint_step_rad must be positive");

  std::string urdf_xml;
  if (!rdf_loader::RDFLoader::loadXmlFileToString(urdf_xml, inputs.at("urdf_xacro").absolute_path.string(), {}))
    throw std::runtime_error("failed to expand the URDF xacro");
  const std::string srdf_xml = readTextFile(inputs.at("candidate_srdf").absolute_path);
  rdf_loader::RDFLoader rdf(urdf_xml, srdf_xml);
  if (!rdf.getURDF() || !rdf.getSRDF())
    throw std::runtime_error("URDF/SRDF parsing did not produce valid models");
  auto robot_model = std::make_shared<moveit::core::RobotModel>(rdf.getURDF(), rdf.getSRDF());
  planning_scene::PlanningScene scene(robot_model);

  const YAML::Node model_contract = requireNode(audit_config, "model_contract");
  const std::size_t expected_links = requireNode(model_contract, "expected_link_count").as<std::size_t>();
  const std::size_t expected_collision_links =
      requireNode(model_contract, "expected_collision_link_count").as<std::size_t>();
  if (robot_model->getLinkModelCount() != expected_links ||
      robot_model->getLinkModelNamesWithCollisionGeometry().size() != expected_collision_links)
    throw std::runtime_error("robot model link counts do not match the audited contract");
  const std::string required_detector = requireNode(model_contract, "required_collision_detector").as<std::string>();
  if (scene.getCollisionDetectorName() != required_detector)
    throw std::runtime_error("required collision detector is " + required_detector + ", active detector is " +
                             scene.getCollisionDetectorName());

  std::size_t never_count = 0;
  std::size_t adjacent_count = 0;
  for (const srdf::Model::CollisionPair& pair : rdf.getSRDF()->getDisabledCollisionPairs())
  {
    if (pair.reason_ == "Never")
      ++never_count;
    else if (pair.reason_ == "Adjacent")
      ++adjacent_count;
    else
      throw std::runtime_error("unexpected disabled-collision reason in candidate SRDF: " + pair.reason_);
  }
  if (never_count != requireNode(model_contract, "expected_never_pair_count").as<std::size_t>() ||
      adjacent_count != requireNode(model_contract, "expected_adjacent_pair_count").as<std::size_t>())
    throw std::runtime_error("candidate SRDF disabled-collision reason counts do not match the contract");

  collision_detection::AllowedCollisionMatrix candidate_acm = scene.getAllowedCollisionMatrix();
  collision_detection::AllowedCollisionMatrix strict_acm = candidate_acm;
  const YAML::Node collision_policy = requireNode(audit_config, "collision_policy");
  if (!requireNode(collision_policy, "reenable_all_never_pairs_for_strict_audit").as<bool>())
    throw std::runtime_error("strict audit must re-enable all reason=Never pairs");
  for (const srdf::Model::CollisionPair& pair : rdf.getSRDF()->getDisabledCollisionPairs())
    if (pair.reason_ == "Never")
    {
      strict_acm.setEntry(pair.link1_, pair.link2_, false);
      collision_detection::AllowedCollision::Type restored_type;
      if (!strict_acm.getEntry(pair.link1_, pair.link2_, restored_type) ||
          restored_type != collision_detection::AllowedCollision::NEVER)
        throw std::runtime_error("failed to restore a reason=Never pair to collision checking: " + pair.link1_ +
                                 " <-> " + pair.link2_);
    }

  const std::string loose_world_id = requireNode(collision_policy, "loose_endpoint_world_id").as<std::string>();
  const std::string attached_id = requireNode(collision_policy, "attached_endpoint_id").as<std::string>();
  const std::string attachment_link = requireNode(collision_policy, "attachment_link").as<std::string>();
  if (!robot_model->hasLinkModel(attachment_link))
    throw std::runtime_error("attachment link is absent from robot model: " + attachment_link);
  std::vector<std::string> touch_links;
  for (const YAML::Node& item : requireNode(collision_policy, "intentional_touch_links"))
  {
    const std::string link = item.as<std::string>();
    if (!robot_model->hasLinkModel(link))
      throw std::runtime_error("intentional touch link is absent from robot model: " + link);
    touch_links.push_back(link);
  }
  if (touch_links.size() != 8U || std::set<std::string>(touch_links.begin(), touch_links.end()).size() != 8U)
    throw std::runtime_error("intentional touch link contract must contain exactly eight unique finger links");

  collision_detection::AllowedCollisionMatrix candidate_contact_acm = candidate_acm;
  collision_detection::AllowedCollisionMatrix strict_contact_acm = strict_acm;
  collision_detection::AllowedCollisionMatrix candidate_attached_acm = candidate_acm;
  collision_detection::AllowedCollisionMatrix strict_attached_acm = strict_acm;
  for (const std::string& link : touch_links)
  {
    candidate_contact_acm.setEntry(loose_world_id, link, true);
    strict_contact_acm.setEntry(loose_world_id, link, true);
  }

  const YAML::Node scene_proxy = requireNode(audit_config, "scene_proxy");
  const Eigen::Vector3d table_center = vector3Of(tabletop_yaml["table"]["center_m"], "table.center_m");
  const Eigen::Vector3d table_size = vector3Of(tabletop_yaml["table"]["size_m"], "table.size_m");
  const Eigen::Vector3d fixture_center =
      vector3Of(tabletop_yaml["fixed_endpoint"]["fixture_center_m"], "fixed_endpoint.fixture_center_m");
  const Eigen::Vector3d fixture_size =
      vector3Of(tabletop_yaml["fixed_endpoint"]["fixture_size_m"], "fixed_endpoint.fixture_size_m");
  const Eigen::Vector3d receptacle_origin =
      vector3Of(tabletop_yaml["fixed_endpoint"]["receptacle_origin_m"], "fixed_endpoint.receptacle_origin_m");
  const Eigen::Vector3d loose_initial = vector3Of(
      tabletop_yaml["loose_endpoint"][is_d38999 ? "initial_origin_m" : "initial_center_m"],
      is_d38999 ? "loose_endpoint.initial_origin_m" : "loose_endpoint.initial_center_m");
  const double body_bottom_offset =
      finiteDouble(tabletop_yaml["loose_endpoint"]["body_bottom_offset_m"], "body_bottom_offset_m");
  const double table_top = table_center.z() + 0.5 * table_size.z();
  Eigen::Vector3d settled_root(loose_initial.x(), loose_initial.y(),
                               is_d38999 ? table_top - body_bottom_offset : table_top + body_bottom_offset);

  const std::string table_world_id = requireNode(scene_proxy, "table_world_id").as<std::string>();
  const std::string fixture_world_id = requireNode(scene_proxy, "fixture_world_id").as<std::string>();
  const std::string fixed_world_id = requireNode(scene_proxy, "fixed_endpoint_world_id").as<std::string>();
  candidate_attached_acm.setEntry(attached_id, table_world_id, true);
  strict_attached_acm.setEntry(attached_id, table_world_id, true);
  scene.getWorldNonConst()->addToObject(
      table_world_id, translated(table_center),
      std::make_shared<shapes::Box>(table_size.x(), table_size.y(), table_size.z()), Eigen::Isometry3d::Identity());
  scene.getWorldNonConst()->addToObject(
      fixture_world_id, translated(fixture_center),
      std::make_shared<shapes::Box>(fixture_size.x(), fixture_size.y(), fixture_size.z()),
      Eigen::Isometry3d::Identity());

  ProxyGeometry proxy;
  if (is_d38999)
  {
    const YAML::Node plug = requireNode(geometry_yaml, "plug");
    const YAML::Node receptacle = requireNode(geometry_yaml, "receptacle");
    const double overall_length = finiteDouble(plug["overall_length"], "plug.overall_length");
    const double rear_radius = finiteDouble(plug["rear_body_radius"], "plug.rear_body_radius");
    const double rear_length = finiteDouble(plug["rear_body_length"], "plug.rear_body_length");
    const double mating_radius =
        finiteDouble(plug["mating_shell_outer_radius"], "plug.mating_shell_outer_radius");
    const double mating_length = finiteDouble(plug["mating_shell_length"], "plug.mating_shell_length");
    const double nut_radius =
        finiteDouble(plug["coupling_nut_outer_radius"], "plug.coupling_nut_outer_radius");
    const double nut_length = finiteDouble(plug["coupling_nut_length"], "plug.coupling_nut_length");
    proxy.loose_shapes = { std::make_shared<shapes::Cylinder>(rear_radius, rear_length),
                           std::make_shared<shapes::Cylinder>(mating_radius, mating_length),
                           std::make_shared<shapes::Cylinder>(nut_radius, nut_length) };
    proxy.loose_shape_poses = {
      translated(Eigen::Vector3d(0.0, 0.0, overall_length - 0.5 * rear_length)),
      translated(Eigen::Vector3d(0.0, 0.0, 0.5 * mating_length)),
      translated(Eigen::Vector3d(0.0, 0.0, 0.5 * overall_length))
    };

    const double flange_side = finiteDouble(receptacle["flange_side"], "receptacle.flange_side");
    const double flange_thickness =
        finiteDouble(receptacle["flange_thickness"], "receptacle.flange_thickness");
    const double shell_radius =
        finiteDouble(receptacle["shell_outer_radius"], "receptacle.shell_outer_radius");
    const double shell_length =
        finiteDouble(receptacle["front_shell_length"], "receptacle.front_shell_length");
    const double fixed_rear_radius =
        finiteDouble(receptacle["rear_body_radius"], "receptacle.rear_body_radius");
    const double fixed_rear_length =
        finiteDouble(receptacle["rear_body_length"], "receptacle.rear_body_length");
    std::vector<shapes::ShapeConstPtr> fixed_shapes = {
      std::make_shared<shapes::Box>(flange_side, flange_side, flange_thickness),
      std::make_shared<shapes::Cylinder>(shell_radius, shell_length),
      std::make_shared<shapes::Cylinder>(fixed_rear_radius, fixed_rear_length)
    };
    EigenSTL::vector_Isometry3d fixed_poses = {
      translated(Eigen::Vector3d(0.0, 0.0, -0.5 * flange_thickness)),
      translated(Eigen::Vector3d(0.0, 0.0, 0.5 * shell_length)),
      translated(Eigen::Vector3d(0.0, 0.0, -0.5 * fixed_rear_length))
    };
    scene.getWorldNonConst()->addToObject(fixed_world_id, translated(receptacle_origin), fixed_shapes, fixed_poses);
  }
  else
  {
    const double nut_radius = finiteDouble(geometry_yaml["coupling_nut_radius"], "coupling_nut_radius");
    const double nut_length = finiteDouble(geometry_yaml["coupling_nut_length"], "coupling_nut_length");
    const double nose_radius = finiteDouble(geometry_yaml["plug_nose_radius"], "plug_nose_radius");
    const double nose_length = finiteDouble(geometry_yaml["plug_nose_length"], "plug_nose_length");
    const double fixed_radius = finiteDouble(geometry_yaml["receptacle_body_radius"], "receptacle_body_radius");
    const double fixed_min_z = finiteDouble(scene_proxy["fixed_endpoint_min_z_offset_m"], "fixed min z");
    const double fixed_max_z = finiteDouble(scene_proxy["fixed_endpoint_max_z_offset_m"], "fixed max z");
    const double fixed_length = fixed_max_z - fixed_min_z;
    if (fixed_length <= 0.0)
      throw std::runtime_error("fixed endpoint proxy z range is invalid");
    const Eigen::Vector3d fixed_center =
        receptacle_origin + Eigen::Vector3d(0.0, 0.0, 0.5 * (fixed_min_z + fixed_max_z));
    scene.getWorldNonConst()->addToObject(fixed_world_id, translated(fixed_center),
                                          std::make_shared<shapes::Cylinder>(fixed_radius, fixed_length),
                                          Eigen::Isometry3d::Identity());

    const double body_radius_scale =
        finiteDouble(scene_proxy["body_radius_scale_from_plug_nose"], "body radius scale");
    const double body_extra_length = finiteDouble(scene_proxy["body_extra_length_m"], "body extra length");
    const double body_center_z = finiteDouble(scene_proxy["body_center_z_offset_m"], "body center z");
    const double nose_center_z = -0.5 * (nut_length + nose_length);
    const double nut_center_z = finiteDouble(scene_proxy["coupling_nut_center_z_offset_m"], "nut center z");
    proxy.loose_shapes = { std::make_shared<shapes::Cylinder>(body_radius_scale * nose_radius,
                                                              nut_length + body_extra_length),
                           std::make_shared<shapes::Cylinder>(nose_radius, nose_length),
                           std::make_shared<shapes::Cylinder>(nut_radius, nut_length) };
    proxy.loose_shape_poses = { translated(Eigen::Vector3d(0.0, 0.0, body_center_z)),
                                translated(Eigen::Vector3d(0.0, 0.0, nose_center_z)),
                                translated(Eigen::Vector3d(0.0, 0.0, nut_center_z)) };
  }
  proxy.loose_world_pose = translated(settled_root);
  scene.getWorldNonConst()->addToObject(loose_world_id, proxy.loose_world_pose, proxy.loose_shapes,
                                        proxy.loose_shape_poses);

  const std::vector<std::string> arm_joint_names = [&robot_yaml]() {
    std::vector<std::string> names;
    for (const YAML::Node& item : robot_yaml["arm_joint_names"])
      names.push_back(item.as<std::string>());
    return names;
  }();
  const std::vector<std::string> hand_joint_names = [&robot_yaml]() {
    std::vector<std::string> names;
    for (const YAML::Node& item : robot_yaml["active_hand_joint_names"])
      names.push_back(item.as<std::string>());
    return names;
  }();
  if (arm_joint_names.size() != 7U || hand_joint_names.size() != 4U)
    throw std::runtime_error("trajectory joint-name contract must contain seven arm and four active hand joints");
  for (const std::string& name : arm_joint_names)
    if (!robot_model->hasJointModel(name))
      throw std::runtime_error("arm joint is absent from robot model: " + name);
  for (const std::string& name : hand_joint_names)
    if (!robot_model->hasJointModel(name))
      throw std::runtime_error("hand joint is absent from robot model: " + name);

  const JointValues home_arm = vectorOf(robot_yaml["home_arm_rad"], 7, "home_arm_rad");
  const JointValues closed_hand(4, 0.0);
  const JointValues open_hand = vectorOf(robot_yaml["open_hand_rad"], 4, "open_hand_rad");
  const JointValues grasp_arm = vectorOf(pick_motion_yaml["grasp_arm_rad"], 7, "grasp_arm_rad");
  const JointValues closure_arm = is_d38999 ?
      vectorOf(pick_motion_yaml["closure_clearance_arm_rad"], 7, "closure_clearance_arm_rad") : grasp_arm;
  const JointValues grasp_hand = vectorOf(pick_motion_yaml["grasp_hand_rad"], 4, "grasp_hand_rad");

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();
  auto setState = [&](const JointValues& arm, const JointValues& hand) {
    for (std::size_t index = 0; index < arm_joint_names.size(); ++index)
      state.setJointPositions(arm_joint_names[index], &arm[index]);
    for (std::size_t index = 0; index < hand_joint_names.size(); ++index)
      state.setJointPositions(hand_joint_names[index], &hand[index]);
    state.updateCollisionBodyTransforms();
  };

  PolicyStats candidate_stats;
  candidate_stats.name = "candidate";
  PolicyStats strict_stats;
  strict_stats.name = "strict_never_reenabled";
  const double minimum_clearance = finiteDouble(collision_policy["minimum_forbidden_distance_m"],
                                                "minimum_forbidden_distance_m");
  const double minimum_robot_table_clearance =
      finiteDouble(collision_policy["minimum_robot_link_table_distance_m"],
                   "minimum_robot_link_table_distance_m");
  if (minimum_clearance < 0.0)
    throw std::runtime_error("minimum_forbidden_distance_m cannot be negative");
  if (minimum_robot_table_clearance < 0.0)
    throw std::runtime_error("minimum_robot_link_table_distance_m cannot be negative");
  const std::size_t maximum_examples =
      requireNode(collision_policy, "maximum_reported_collision_examples_per_policy").as<std::size_t>();
  const std::vector<std::string> world_object_ids = { table_world_id, fixture_world_id, fixed_world_id, loose_world_id };
  auto robotTableAcm = [&](const collision_detection::AllowedCollisionMatrix& base) {
    collision_detection::AllowedCollisionMatrix result = base;
    for (const std::string& link : robot_model->getLinkModelNamesWithCollisionGeometry())
    {
      for (const std::string& object_id : world_object_ids)
        result.setEntry(link, object_id, object_id != table_world_id);
      result.setEntry(link, attached_id, true);
    }
    for (const std::string& object_id : world_object_ids)
      result.setEntry(attached_id, object_id, true);
    return result;
  };
  std::size_t sample_index = 0;
  std::size_t bounds_violation_samples = 0;
  std::map<std::string, std::size_t> phase_counts;
  JointValues previous_command;
  bool have_previous_command = false;
  double observed_maximum_joint_step = 0.0;
  std::string maximum_joint_step_phase;
  std::size_t maximum_joint_step_sample = 0;

  auto inspectSample = [&](const std::string& phase, const JointValues& arm, const JointValues& hand,
                           const SceneMode mode) {
    setState(arm, hand);
    JointValues command = arm;
    command.insert(command.end(), hand.begin(), hand.end());
    if (have_previous_command)
    {
      for (std::size_t index = 0; index < command.size(); ++index)
      {
        const double step = std::abs(command[index] - previous_command[index]);
        if (step > observed_maximum_joint_step)
        {
          observed_maximum_joint_step = step;
          maximum_joint_step_phase = phase;
          maximum_joint_step_sample = sample_index;
        }
      }
    }
    previous_command = command;
    have_previous_command = true;
    if (!state.satisfiesBounds(0.0))
      ++bounds_violation_samples;

    const auto& selected_candidate_acm = mode == SceneMode::INTENTIONAL_FINGER_CONTACT ? candidate_contact_acm :
                                         mode == SceneMode::ATTACHED ? candidate_attached_acm : candidate_acm;
    const auto& selected_strict_acm = mode == SceneMode::INTENTIONAL_FINGER_CONTACT ? strict_contact_acm :
                                      mode == SceneMode::ATTACHED ? strict_attached_acm : strict_acm;
    const auto candidate_table_acm = robotTableAcm(selected_candidate_acm);
    const auto strict_table_acm = robotTableAcm(selected_strict_acm);
    const std::size_t phase_sample_index = phase_counts[phase];
    inspectPolicy(scene, state, selected_candidate_acm, phase, sample_index, phase_sample_index, minimum_clearance,
                  maximum_examples, candidate_table_acm, candidate_stats);
    inspectPolicy(scene, state, selected_strict_acm, phase, sample_index, phase_sample_index, minimum_clearance,
                  maximum_examples, strict_table_acm, strict_stats);
    ++phase_counts[phase];
    ++sample_index;
  };

  auto runPhase = [&](const std::string& name, const std::size_t steps, const JointValues& arm_start,
                      const JointValues& arm_target, const JointValues& hand_start, const JointValues& hand_target,
                      const bool interpolate_arm, const bool interpolate_hand, const SceneMode mode) {
    for (std::size_t index = 0; index < steps; ++index)
    {
      const double fraction = static_cast<double>(index + 1U) / static_cast<double>(steps);
      const double blend = minimumJerkBlend(fraction);
      inspectSample(name, interpolate_arm ? interpolate(arm_start, arm_target, blend) : arm_target,
                    interpolate_hand ? interpolate(hand_start, hand_target, blend) : hand_target, mode);
    }
  };

  JointValues current_arm = home_arm;
  JointValues current_hand = closed_hand;
  runPhase("initial_settle", exactSteps(finiteDouble(tabletop_yaml["physics"]["settle_duration_s"],
                                                     "settle_duration_s"),
                                        rate_hz, "initial_settle"),
           current_arm, current_arm, current_hand, current_hand, false, false, SceneMode::PRECLOSURE);
  runPhase("home_hand_open", exactSteps(finiteDouble(home_motion_yaml["hand_open_duration_s"],
                                                     "hand_open_duration_s"),
                                        rate_hz, "home_hand_open"),
           current_arm, current_arm, current_hand, open_hand, false, true, SceneMode::PRECLOSURE);
  current_hand = open_hand;

  for (const YAML::Node& segment : approach_segments_yaml)
  {
    const std::string name = requireNode(segment, "name").as<std::string>();
    const JointValues target = vectorOf(segment["target_arm_rad"], 7, name + ".target_arm_rad");
    runPhase(name, exactSteps(finiteDouble(segment["duration_s"], name + ".duration_s"), rate_hz, name), current_arm,
             target, current_hand, current_hand, true, false, SceneMode::PRECLOSURE);
    current_arm = target;
  }
  const JointValues pregrasp_arm = current_arm;
  runPhase("pregrasp_hold", exactSteps(pregrasp_hold_duration, rate_hz, "pregrasp_hold"),
           current_arm, current_arm, current_hand, current_hand, false, false, SceneMode::PRECLOSURE);
  runPhase("open_hand_descent",
           exactSteps(finiteDouble(pick_motion_yaml["descent_duration_s"], "descent_duration_s"), rate_hz,
                      "open_hand_descent"),
           current_arm, closure_arm, current_hand, current_hand, true, false, SceneMode::PRECLOSURE);
  current_arm = closure_arm;
  runPhase("open_grasp_tare",
           exactSteps(finiteDouble(pick_motion_yaml["open_tare_duration_s"], "open_tare_duration_s"), rate_hz,
                      "open_grasp_tare"),
           current_arm, current_arm, current_hand, current_hand, false, false, SceneMode::PRECLOSURE);
  runPhase("physical_hand_closure",
           exactSteps(finiteDouble(pick_motion_yaml["closure_duration_s"], "closure_duration_s"), rate_hz,
                      "physical_hand_closure"),
           current_arm, current_arm, current_hand, grasp_hand, false, true, SceneMode::INTENTIONAL_FINGER_CONTACT);
  current_hand = grasp_hand;
  if (is_d38999)
  {
    runPhase("closed_hand_seating",
             exactSteps(finiteDouble(pick_motion_yaml["closed_seating_duration_s"], "closed_seating_duration_s"),
                        rate_hz, "closed_hand_seating"),
             current_arm, grasp_arm, current_hand, current_hand, true, false,
             SceneMode::INTENTIONAL_FINGER_CONTACT);
    current_arm = grasp_arm;
  }
  runPhase("physical_grip_preload",
           exactSteps(finiteDouble(pick_motion_yaml["preload_duration_s"], "preload_duration_s"), rate_hz,
                      "physical_grip_preload"),
           current_arm, current_arm, current_hand, current_hand, false, false,
           SceneMode::INTENTIONAL_FINGER_CONTACT);

  const std::map<std::string, std::vector<std::string>> finger_groups = {
    { "finger_1", { "f1Link1", "f1Link2", "f1Link3" } },
    { "finger_2", { "f2Link1", "f2Link2" } },
    { "finger_3", { "f3Link1", "f3Link2", "f3Link3" } }
  };
  std::map<std::string, MinimumRecord> final_finger_object_distances;
  std::size_t final_touching_finger_count = 0U;
  setState(current_arm, current_hand);
  for (const auto& group : finger_groups)
  {
    collision_detection::AllowedCollisionMatrix isolated = strict_acm;
    for (const std::string& link : robot_model->getLinkModelNamesWithCollisionGeometry())
    {
      for (const std::string& object_id : world_object_ids)
        isolated.setEntry(link, object_id, true);
      isolated.setEntry(link, attached_id, true);
    }
    for (const std::string& link : group.second)
      isolated.setEntry(link, loose_world_id, false);
    collision_detection::DistanceRequest request;
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::GLOBAL;
    request.acm = &isolated;
    request.enableGroup(robot_model);
    collision_detection::DistanceResult result;
    scene.getCollisionEnv()->distanceRobot(request, result, state);
    MinimumRecord record;
    updateMinimum(record, result, "physical_grip_preload", sample_index - 1U,
                  phase_counts["physical_grip_preload"] - 1U);
    if (!record.available)
      throw std::runtime_error("final finger-to-object proxy distance is unavailable for " + group.first);
    const bool pair_matches =
        (record.links[0] == loose_world_id &&
         std::find(group.second.begin(), group.second.end(), record.links[1]) != group.second.end()) ||
        (record.links[1] == loose_world_id &&
         std::find(group.second.begin(), group.second.end(), record.links[0]) != group.second.end());
    if (!pair_matches)
      throw std::runtime_error("final finger-to-object proxy returned an unexpected pair for " + group.first);
    if (record.distance <= 0.0)
      ++final_touching_finger_count;
    final_finger_object_distances.emplace(group.first, record);
  }
  const bool final_three_finger_contact_reachable = final_touching_finger_count == 3U;

  setState(current_arm, current_hand);
  const Eigen::Isometry3d world_to_tcp = state.getGlobalLinkTransform(attachment_link);
  const Eigen::Isometry3d tcp_to_object = world_to_tcp.inverse() * proxy.loose_world_pose;
  if (!scene.getWorldNonConst()->removeObject(loose_world_id))
    throw std::runtime_error("loose endpoint disappeared before the attachment transition");
  state.attachBody(attached_id, tcp_to_object, proxy.loose_shapes, proxy.loose_shape_poses, touch_links, attachment_link);
  state.updateCollisionBodyTransforms();

  runPhase("physical_grip_lift",
           exactSteps(finiteDouble(pick_motion_yaml["lift_duration_s"], "lift_duration_s"), rate_hz,
                      "physical_grip_lift"),
           current_arm, pregrasp_arm, current_hand, current_hand, true, false, SceneMode::ATTACHED);
  current_arm = pregrasp_arm;
  runPhase("unsupported_final_hold",
           exactSteps(finiteDouble(pick_motion_yaml["final_hold_duration_s"], "final_hold_duration_s"), rate_hz,
                      "unsupported_final_hold"),
           current_arm, current_arm, current_hand, current_hand, false, false, SceneMode::ATTACHED);

  const bool sampling_complete = sample_index == expected_sample_count;
  const bool bounded_sampling = observed_maximum_joint_step <= maximum_joint_step;
  const bool candidate_discrete_clear = candidate_stats.self_collision_samples == 0U &&
                                        candidate_stats.environment_collision_samples == 0U &&
                                        candidate_stats.self_clearance_violations == 0U &&
                                        candidate_stats.environment_clearance_violations == 0U;
  const bool strict_discrete_clear = strict_stats.self_collision_samples == 0U &&
                                     strict_stats.environment_collision_samples == 0U &&
                                     strict_stats.self_clearance_violations == 0U &&
                                     strict_stats.environment_clearance_violations == 0U;
  const bool candidate_robot_table_margin_clear = candidate_stats.minimum_robot_link_table_distance.available &&
      candidate_stats.minimum_robot_link_table_distance.distance >= minimum_robot_table_clearance;
  const bool strict_robot_table_margin_clear = strict_stats.minimum_robot_link_table_distance.available &&
      strict_stats.minimum_robot_link_table_distance.distance >= minimum_robot_table_clearance;
  const bool continuous_collision_verified = false;
  const bool require_candidate =
      requireNode(collision_policy, "require_candidate_discrete_collision_free").as<bool>();
  const bool require_strict = requireNode(collision_policy, "require_strict_discrete_collision_free").as<bool>();
  const bool require_continuous = requireNode(collision_policy, "require_continuous_collision_verified").as<bool>();
  const bool passed = sampling_complete && bounded_sampling && bounds_violation_samples == 0U &&
                      (!require_candidate || (candidate_discrete_clear && candidate_robot_table_margin_clear)) &&
                      (!require_strict || (strict_discrete_clear && strict_robot_table_margin_clear)) &&
                      (!require_continuous || continuous_collision_verified);

  std::string status = "PASSED";
  if (!sampling_complete || !bounded_sampling || bounds_violation_samples != 0U)
    status = "FAIL_CLOSED_SAMPLING_OR_BOUNDS";
  else if ((require_candidate && (!candidate_discrete_clear || !candidate_robot_table_margin_clear)) ||
           (require_strict && (!strict_discrete_clear || !strict_robot_table_margin_clear)))
    status = "FAIL_CLOSED_DISCRETE_COLLISION_OR_CLEARANCE";
  else if (require_continuous && !continuous_collision_verified)
    status = "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED";

  YAML::Node report;
  report["schema_version"] = kReportSchemaVersion;
  report["profile"] = profile;
  report["status"] = status;
  report["passed"] = passed;
  report["project_root"] = arguments.project_root.string();
  report["audit_config"] = arguments.config.string();
  for (const auto& entry : inputs)
  {
    YAML::Node item;
    item["path"] = entry.second.relative_path.string();
    item["expected_sha256"] = entry.second.expected_sha256;
    item["actual_sha256"] = entry.second.actual_sha256;
    item["hash_matches"] = entry.second.expected_sha256 == entry.second.actual_sha256;
    report["inputs"][entry.first] = item;
  }
  report["model"]["name"] = robot_model->getName();
  report["model"]["link_count"] = robot_model->getLinkModelCount();
  report["model"]["collision_link_count"] = robot_model->getLinkModelNamesWithCollisionGeometry().size();
  report["model"]["collision_detector"] = scene.getCollisionDetectorName();
  report["model"]["candidate_never_pair_count"] = never_count;
  report["model"]["candidate_adjacent_pair_count"] = adjacent_count;
  report["model"]["strict_reenabled_never_pair_count"] = never_count;
  report["scene_proxy"]["world_objects_before_attachment"].push_back(
      table_world_id);
  report["scene_proxy"]["world_objects_before_attachment"].push_back(
      fixture_world_id);
  report["scene_proxy"]["world_objects_before_attachment"].push_back(
      fixed_world_id);
  report["scene_proxy"]["world_objects_before_attachment"].push_back(loose_world_id);
  report["scene_proxy"]["free_endpoint_proxy_is_conservative_solid_overapproximation"] = true;
  report["scene_proxy"]["fixed_endpoint_proxy_is_conservative_solid_overapproximation"] = true;
  report["scene_proxy"]["loose_endpoint_settled_root_world_m"].push_back(settled_root.x());
  report["scene_proxy"]["loose_endpoint_settled_root_world_m"].push_back(settled_root.y());
  report["scene_proxy"]["loose_endpoint_settled_root_world_m"].push_back(settled_root.z());
  report["scene_proxy"]["attachment_is_offline_rigid_grasp_assumption"] = true;
  report["scene_proxy"]["attached_endpoint_table_contact_excluded_during_lift"] = true;
  report["scene_proxy"]["attached_endpoint_table_exclusion_reason"] =
      "The object starts lift supported by the table. This expected object-table transition is not a robot-link/table "
      "collision; robot-link/table checks remain enabled and are reported separately.";
  report["scene_proxy"]["attachment_link"] = attachment_link;
  for (const std::string& link : touch_links)
    report["scene_proxy"]["intentional_touch_links"].push_back(link);
  report["final_object_contact_proxy"]["evaluated_phase"] = "physical_grip_preload";
  report["final_object_contact_proxy"]["touch_definition"] = "signed_distance_m <= 0";
  report["final_object_contact_proxy"]["touching_finger_count"] = final_touching_finger_count;
  report["final_object_contact_proxy"]["three_finger_contact_reachable"] = final_three_finger_contact_reachable;
  report["final_object_contact_proxy"]["exact_nonpenetrating_static_contact_verified"] = false;
  report["final_object_contact_proxy"]["interpretation"] =
      "Negative signed distance is a torque-limited preload target, not a realizable exact rigid-body position. "
      "Physics must establish finite equilibrium at the contact surface.";
  for (const auto& entry : final_finger_object_distances)
  {
    report["final_object_contact_proxy"]["fingers"][entry.first] = minimumNode(entry.second);
    report["final_object_contact_proxy"]["fingers"][entry.first]["target_contact_reachable"] =
        entry.second.distance <= 0.0;
  }
  report["sampling"]["rate_hz"] = rate_hz;
  report["sampling"]["expected_sample_count"] = expected_sample_count;
  report["sampling"]["checked_sample_count"] = sample_index;
  report["sampling"]["complete"] = sampling_complete;
  report["sampling"]["maximum_allowed_joint_step_rad"] = maximum_joint_step;
  report["sampling"]["observed_maximum_joint_step_rad"] = observed_maximum_joint_step;
  report["sampling"]["maximum_joint_step_phase"] = maximum_joint_step_phase;
  report["sampling"]["maximum_joint_step_sample_index"] = maximum_joint_step_sample;
  report["sampling"]["bounded"] = bounded_sampling;
  report["sampling"]["joint_bounds_violation_samples"] = bounds_violation_samples;
  for (const auto& phase : phase_counts)
    report["sampling"]["phase_sample_counts"][phase.first] = phase.second;
  report["discrete_collision"]["candidate"] = policyNode(candidate_stats);
  report["discrete_collision"]["candidate"]["collision_free"] = candidate_discrete_clear;
  report["discrete_collision"]["candidate"]["required_robot_link_table_distance_m"] =
      minimum_robot_table_clearance;
  report["discrete_collision"]["candidate"]["robot_link_table_margin_clear"] =
      candidate_robot_table_margin_clear;
  report["discrete_collision"]["strict_never_reenabled"] = policyNode(strict_stats);
  report["discrete_collision"]["strict_never_reenabled"]["collision_free"] = strict_discrete_clear;
  report["discrete_collision"]["strict_never_reenabled"]["required_robot_link_table_distance_m"] =
      minimum_robot_table_clearance;
  report["discrete_collision"]["strict_never_reenabled"]["robot_link_table_margin_clear"] =
      strict_robot_table_margin_clear;
  report["continuous_collision"]["verified"] = continuous_collision_verified;
  report["continuous_collision"]["backend"] = scene.getCollisionDetectorName();
  report["continuous_collision"]["self_collision_verified"] = false;
  report["continuous_collision"]["robot_world_verified"] = false;
  report["continuous_collision"]["reason"] =
      "MoveIt 2 Humble FCL reports continuous collision as not implemented; no combined continuous self/world/distance "
      "gate is available. Discrete 240 Hz samples are not a continuous proof.";
  report["limitations"]["physics_contact_or_grasp_verified"] = false;
  report["limitations"]["world_world_collision_checked"] = false;
  report["limitations"]["mesh_fidelity_to_flight_connector_verified"] = false;
  report["limitations"]["candidate_srdf_activated"] = false;

  writeReport(report, arguments);
  return passed || arguments.report_only ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    return runAudit(parseArguments(argc, argv));
  }
  catch (const std::exception& error)
  {
    YAML::Node failure;
    failure["schema_version"] = kReportSchemaVersion;
    failure["status"] = "FAIL_CLOSED_RUNTIME_ERROR";
    failure["passed"] = false;
    failure["error"] = error.what();
    YAML::Emitter emitter;
    emitter << failure;
    std::cerr << emitter.c_str() << '\n';
    return 1;
  }
}
