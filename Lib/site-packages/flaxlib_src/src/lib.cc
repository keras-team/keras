// Copyright 2024 The Flax Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/optional.h>
#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <Python.h>

namespace nb = nanobind;
using namespace nb::literals;

// -----------------------------------
// helper functions
// -----------------------------------
intptr_t nb_id(const nb::object &obj)
{
  // Get the object ID
  return reinterpret_cast<intptr_t>(obj.ptr());
}

nb::tuple vector_to_tuple(const std::vector<nb::object> &vec)
{

  if (vec.empty())
  {
    return nb::tuple();
  }
  else
  {
    return nb::tuple(nb::cast(vec));
  }
}

// 1. Hash function for nb::object
struct NbObjectHash
{
  std::size_t operator()(const nb::object &obj) const
  {
    return nb::hash(obj);
  }
};

// 2. Equality function for nb::object (Important!)
struct NbObjectEqual
{
  bool operator()(const nb::object &a, const nb::object &b) const
  {
    return a.equal(b);
  }
};

NB_MAKE_OPAQUE(std::unordered_map<nb::object, int, NbObjectHash, NbObjectEqual>);

namespace flaxlib
{
  //---------------------------------------------------------------
  // RefMap
  //---------------------------------------------------------------

  using RefMap = std::unordered_map<nb::object, int, NbObjectHash, NbObjectEqual>;

  std::optional<int> ref_map_get(RefMap &map, nb::object &key, std::optional<int> default_value = std::nullopt)
  {
    auto it = map.find(key);
    if (it != map.end())
    {
      return it->second;
    }
    else
    {
      return std::nullopt;
    }
  }

  //---------------------------------------------------------------
  // NNXContext
  //---------------------------------------------------------------

  struct PythonContext
  {
    nb::object nnx;
    nb::object graph;
    nb::object jax;
    nb::object np;
    nb::object jax_Array;
    nb::object np_ndarray;
    nb::type_object GraphNodeImpl;
    nb::type_object PytreeNodeImpl;
    nb::type_object Object;
    nb::type_object Variable;
    nb::object get_node_impl;

    PythonContext()
    {
      nnx = nb::module_::import_("flax.nnx");
      graph = nb::module_::import_("flax.nnx.graph");
      jax = nb::module_::import_("jax");
      np = nb::module_::import_("numpy");
      jax_Array = jax.attr("Array");
      np_ndarray = np.attr("ndarray");
      GraphNodeImpl = graph.attr("GraphNodeImpl");
      PytreeNodeImpl = graph.attr("PytreeNodeImpl");
      Object = nnx.attr("Object");
      Variable = graph.attr("Variable");
      get_node_impl = graph.attr("get_node_impl");
    }

    ~PythonContext()
    {
      graph.release();
      jax.release();
      np.release();
      jax_Array.release();
      np_ndarray.release();
      GraphNodeImpl.release();
      PytreeNodeImpl.release();
      Variable.release();
      get_node_impl.release();
    }
  };

  static std::optional<PythonContext> _python_context;

  PythonContext &get_python_context()
  {
    if (!_python_context)
    {
      _python_context.emplace();
    }
    return *_python_context;
  }

  //---------------------------------------------------------------
  // fingerprint
  //---------------------------------------------------------------
  std::tuple<nb::object, nb::object> _key_values_metadata(
      PythonContext &ctx,
      nb::object &node,
      nb::object &node_impl)
  {
    if (nb::isinstance(node, ctx.Object))
    {
      nb::dict nodes_dict = node.attr("__dict__");
      nb::handle object_state = nodes_dict["_object__state"];
      nb::del(nodes_dict["_object__state"]);
      auto nodes = nodes_dict.items();
      nodes.sort();
      nodes_dict["_object__state"] = object_state;
      auto metadata = nb::make_tuple(node.type(), object_state.attr("_initializing"));
      return {nodes, metadata};
    }
    else if (PyList_Check(node.ptr()) || PyTuple_Check(node.ptr()))
    {
      int i = 0;
      nb::list values;
      for (const auto &value : node)
      {
        values.append(nb::make_tuple(i, value));
        i += 1;
      }
      return {values, nb::none()};
    }
    else
    {
      auto values_metadata = node_impl.attr("flatten")(node);
      auto values = values_metadata[0];
      auto metadata = values_metadata[1];
      return {values, metadata};
    }
  }

  nb::tuple _graph_fingerprint_recursive(
      PythonContext &ctx,
      nb::object &node,
      nb::object &node_impl,
      RefMap &ref_index,
      RefMap &new_ref_index,
      int &next_index)
  {
    bool is_pytree_node = node_impl.type().is(ctx.PytreeNodeImpl);
    bool is_graph_node = node_impl.type().is(ctx.GraphNodeImpl);

    if (is_pytree_node)
    {
      // pass
    }
    else if (ref_index.find(node) != ref_index.end())
    {
      return nb::make_tuple(nb_id(node), node.type(), ref_index[node]);
    }
    else if (new_ref_index.find(node) != new_ref_index.end())
    {
      return nb::make_tuple(nb_id(node), node.type(), new_ref_index[node]);
    }

    // only cache graph nodes
    int index;
    if (is_graph_node)
    {
      index = new_ref_index[node] = next_index;
      next_index += 1;
    }
    else
    {
      index = -1;
    }

    std::vector<nb::object> attributes;

    auto [values, metadata] = _key_values_metadata(ctx, node, node_impl);

    for (const auto &key_value : values)
    {
      nb::object key = key_value[0];
      nb::object value = key_value[1];
      auto value_node_impl = ctx.get_node_impl(value);
      if (!value_node_impl.is_none())
      {
        auto node_fp = _graph_fingerprint_recursive(ctx, value, value_node_impl, ref_index, new_ref_index, next_index);
        attributes.push_back(nb::make_tuple(key, node_fp));
      }
      else if (nb::isinstance(value, ctx.Variable))
      {
        if (ref_index.find(value) != ref_index.end())
        {
          attributes.push_back(nb::make_tuple(key, nb_id(value), value.type(), ref_index[value]));
        }
        else if (new_ref_index.find(value) != new_ref_index.end())
        {
          attributes.push_back(nb::make_tuple(key, nb_id(value), value.type(), new_ref_index[value]));
        }
        else
        {
          auto variable_index = new_ref_index[value] = next_index;
          next_index += 1;
          auto var_meta = nb::tuple(value.attr("_var_metadata").attr("items")());
          attributes.push_back(nb::make_tuple(key, nb_id(value), value.type(), variable_index, var_meta));
        }
      }
      else // static attribute
      {
        if (nb::isinstance(value, ctx.jax_Array) || nb::isinstance(value, ctx.np_ndarray))
        {
          auto repr = "Arrays leaves are not supported: " + nb::cast<std::string>(nb::repr(value));
        }
        attributes.push_back(nb::make_tuple(key, value));
      }
    }

    auto node_fp = nb::make_tuple(
        is_graph_node ? nb::cast(nb_id(node)) : nb::none(),
        node_impl.attr("type"),
        index,
        vector_to_tuple(attributes),
        metadata);

    return node_fp;
  }

  nb::tuple _graph_fingerprint(
      nb::object &node,
      nb::object &node_impl,
      RefMap &ref_index,
      RefMap &new_ref_index,
      int next_index)
  {
    auto ctx = get_python_context();
    auto node_fp = _graph_fingerprint_recursive(ctx, node, node_impl, ref_index, new_ref_index, next_index);
    return nb::make_tuple(node_fp, next_index);
  }

  NB_MODULE(flaxlib_cpp, m)
  {
    // Remove the conflicting binding
    nb::bind_map<RefMap>(m, "RefMap")
        .def("get", &ref_map_get, nb::arg("key").none(), nb::arg("default_value").none());
    m.def("_graph_fingerprint", &_graph_fingerprint);
  }
} // namespace flaxlib