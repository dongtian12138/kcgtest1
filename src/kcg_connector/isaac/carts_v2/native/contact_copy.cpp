// Read-only bulk access to the existing PhysX ContactDataVector.
// The layout below is the public NVIDIA ContactData definition:
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/ovruntime/include/omni/physx/ContactEvent.h
// We use the installed Carbonite Float3 type, and check the registered ABI before
// accessing the vector. All output numbers and containers are owned by Python.
#include <carb/Types.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace omni { namespace physx {
struct ContactData {
    carb::Float3 position;
    carb::Float3 normal;
    carb::Float3 impulse;
    float separation;
    uint32_t faceIndex0;
    uint32_t faceIndex1;
    uint64_t material0;
    uint64_t material1;
};
}}
using Contacts = std::vector<omni::physx::ContactData>;
PYBIND11_MAKE_OPAQUE(Contacts);
namespace py = pybind11;
static_assert(sizeof(omni::physx::ContactData) == 64);
static_assert(offsetof(omni::physx::ContactData, separation) == 36);
static_assert(offsetof(omni::physx::ContactData, material0) == 48);

static py::tuple vector_copy(const carb::Float3& value) {
    py::tuple result(3);
    result[0] = py::float_(value.x);
    result[1] = py::float_(value.y);
    result[2] = py::float_(value.z);
    return result;
}

static py::list decode_ranges(py::handle input,
                             const std::vector<std::array<py::ssize_t, 2>>& ranges) {
    auto* info = py::detail::get_type_info(typeid(omni::physx::ContactData), false);
    if (!info || info->type_size != sizeof(omni::physx::ContactData)
              || info->type_align != alignof(omni::physx::ContactData)) {
        throw std::runtime_error("PhysX ContactData ABI is unavailable or incompatible");
    }
    const auto& data = py::cast<const Contacts&>(input);
    py::list result(ranges.size());
    py::str position("position_m"), normal("normal"), impulse("impulse_n_s"), separation("separation_m");
    for (size_t g = 0; g < ranges.size(); ++g) {
        const auto start = ranges[g][0], count = ranges[g][1];
        if (start < 0 || count < 0 || static_cast<size_t>(start) > data.size()
            || static_cast<size_t>(count) > data.size() - static_cast<size_t>(start)) {
            throw std::runtime_error("contact header data range is invalid");
        }
        py::list points(count);
        for (py::ssize_t i = 0; i < count; ++i) {
            const auto& p = data[static_cast<size_t>(start + i)];
            py::dict point;
            point[position] = vector_copy(p.position);
            point[normal] = vector_copy(p.normal);
            point[impulse] = vector_copy(p.impulse);
            point[separation] = py::float_(p.separation);
            points[i] = std::move(point);
        }
        result[g] = std::move(points);
    }
    return result;
}

static py::tuple pack_ranges(py::handle input,
                            const std::vector<std::array<py::ssize_t, 2>>& ranges) {
    auto* info = py::detail::get_type_info(typeid(omni::physx::ContactData), false);
    if (!info || info->type_size != sizeof(omni::physx::ContactData)
              || info->type_align != alignof(omni::physx::ContactData)) {
        throw std::runtime_error("PhysX ContactData ABI is unavailable or incompatible");
    }
    const auto& data = py::cast<const Contacts&>(input);
    py::list result(ranges.size());
    bool finite = true;
    for (size_t g = 0; g < ranges.size(); ++g) {
        const auto start = ranges[g][0], count = ranges[g][1];
        if (start < 0 || count < 0 || static_cast<size_t>(start) > data.size()
            || static_cast<size_t>(count) > data.size() - static_cast<size_t>(start)) {
            throw std::runtime_error("contact header data range is invalid");
        }
        py::bytes block = py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr, count*80));
        char* dest = PyBytes_AS_STRING(block.ptr());
        for (py::ssize_t i = 0; i < count; ++i) {
            const auto& p = data[static_cast<size_t>(start + i)];
            const double values[10] = {p.position.x,p.position.y,p.position.z,
                p.normal.x,p.normal.y,p.normal.z,p.impulse.x,p.impulse.y,p.impulse.z,p.separation};
            for (double v : values) finite = finite && std::isfinite(v);
            std::memcpy(dest+i*80,values,80);
        }
        result[g] = std::move(block);
    }
    return py::make_tuple(result,finite);
}

PYBIND11_MODULE(_contact_copy_native, m) {
    m.doc() = "Owned, lossless bulk copies of existing native contact reports";
    m.def("decode_ranges", &decode_ranges);
    m.def("pack_ranges", &pack_ranges);
    m.attr("contact_data_size") = sizeof(omni::physx::ContactData);
    m.attr("pybind_internals_id") = PYBIND11_INTERNALS_ID;
}
