// Read-only native contact/header copying. ABI layout follows NVIDIA ContactEvent.h.
// Each recorder owns its path cache; point values, offsets and counts are never cached.
#include <carb/Types.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <vector>
namespace omni { namespace physx {
struct ContactEventType { enum Enum {eCONTACT_FOUND,eCONTACT_LOST,eCONTACT_PERSIST}; };
struct ContactEventHeader {
    ContactEventType::Enum type; int64_t stageId;
    uint64_t actor0,actor1,collider0,collider1;
    uint32_t contactDataOffset,numContactData,frictionAnchorsDataOffset,numfrictionAnchorsData,protoIndex0,protoIndex1;
};
struct ContactData {
    carb::Float3 position,normal,impulse; float separation;
    uint32_t faceIndex0,faceIndex1; uint64_t material0,material1;
};
}}
using Headers=std::vector<omni::physx::ContactEventHeader>;
using Points=std::vector<omni::physx::ContactData>;
PYBIND11_MAKE_OPAQUE(Headers);
PYBIND11_MAKE_OPAQUE(Points);
namespace py=pybind11;
static_assert(sizeof(omni::physx::ContactEventHeader)==72);
static_assert(offsetof(omni::physx::ContactEventHeader,actor0)==16);
static_assert(offsetof(omni::physx::ContactEventHeader,contactDataOffset)==48);
static_assert(sizeof(omni::physx::ContactData)==64);
using Key=std::array<uint64_t,4>;
struct KeyHash {
    size_t operator()(const Key& v) const noexcept {
        size_t h=0; for(auto x:v) h^=std::hash<uint64_t>{}(x)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); return h;
    }
};
struct Decoder {
    py::object path_decoder,packed_factory;
    bool compact_float32;
    std::unordered_map<Key,py::tuple,KeyHash> paths;
    Decoder(py::object p,py::object f,bool compact=false):path_decoder(p),packed_factory(f),compact_float32(compact){}
    py::tuple get_paths(const omni::physx::ContactEventHeader& h) {
        Key k{h.actor0,h.actor1,h.collider0,h.collider1};
        auto found=paths.find(k);if(found!=paths.end())return found->second;
        py::tuple result(4);for(size_t i=0;i<4;++i)result[i]=py::str(path_decoder(py::int_(k[i])));
        paths.emplace(k,result);return result;
    }
    void check_abi() {
        auto* h=py::detail::get_type_info(typeid(omni::physx::ContactEventHeader),false);
        auto* p=py::detail::get_type_info(typeid(omni::physx::ContactData),false);
        if(!h || !p || h->type_size!=72 || p->type_size!=64 || h->type_align!=8 || p->type_align!=8)
            throw std::runtime_error("Unsupported PhysX contact ABI");
    }
    py::list events_impl(const Headers& headers) {
        py::list out(headers.size());size_t j=0;
        for(const auto& h:headers)out[j++]=py::make_tuple(get_paths(h),h.numContactData);
        return out;
    }
    py::list full_impl(const Headers& headers,const Points& data) {
        py::list out(headers.size());size_t j=0;
        py::str paths_key("paths"),count_key("records"),offset_key("contact_data_offset"),points_key("contacts");
        for(const auto& h:headers) {
            size_t offset=h.contactDataOffset,count=h.numContactData;
            if(offset>data.size() || count>data.size()-offset)throw std::runtime_error("Invalid native contact range");
            size_t stride=compact_float32?40:80;
            py::bytes bytes=py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr,count*stride));
            char* dest=PyBytes_AS_STRING(bytes.ptr());bool finite=true;
            for(size_t i=0;i<count;++i) {
                const auto& p=data[offset+i];
                float v[10]={p.position.x,p.position.y,p.position.z,p.normal.x,p.normal.y,p.normal.z,
                    p.impulse.x,p.impulse.y,p.impulse.z,p.separation};
                for(float x:v)finite=finite&&std::isfinite(x);
                if(compact_float32)std::memcpy(dest+stride*i,v,stride);
                else { double promoted[10];for(size_t k=0;k<10;++k)promoted[k]=v[k];
                       std::memcpy(dest+stride*i,promoted,stride); }
            }
            py::dict row;row[paths_key]=get_paths(h);row[count_key]=h.numContactData;row[offset_key]=h.contactDataOffset;
            row[points_key]=packed_factory(bytes,py::arg("native_finite_verified")=finite);out[j++]=std::move(row);
        }
        return out;
    }
    py::list events(py::handle h) {check_abi();return events_impl(py::cast<const Headers&>(h));}
    py::list full(py::handle h,py::handle p) {check_abi();return full_impl(py::cast<const Headers&>(h),py::cast<const Points&>(p));}
};
PYBIND11_MODULE(_contact_reports_native,m) {
    py::class_<Decoder>(m,"Decoder").def(py::init<py::object,py::object,bool>(),
        py::arg("path_decoder"),py::arg("packed_factory"),py::arg("compact_float32")=false)
        .def("events",&Decoder::events).def("full",&Decoder::full);
    m.attr("header_size")=sizeof(omni::physx::ContactEventHeader);
    m.attr("point_size")=sizeof(omni::physx::ContactData);
    m.attr("pybind_internals_id")=PYBIND11_INTERNALS_ID;
}
