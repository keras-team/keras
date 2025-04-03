// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_attr_value_proto_H_
#define tensorflow_core_framework_attr_value_proto_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.AttrValue.ListValue
string ProtoDebugString(
    const ::tensorflow::AttrValue_ListValue& msg);
string ProtoShortDebugString(
    const ::tensorflow::AttrValue_ListValue& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::AttrValue_ListValue* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.AttrValue
string ProtoDebugString(
    const ::tensorflow::AttrValue& msg);
string ProtoShortDebugString(
    const ::tensorflow::AttrValue& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::AttrValue* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.NameAttrList
string ProtoDebugString(
    const ::tensorflow::NameAttrList& msg);
string ProtoShortDebugString(
    const ::tensorflow::NameAttrList& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::NameAttrList* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_attr_value_proto_H_
