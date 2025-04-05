// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
struct AutoBroadcastSpec;
}

/// \brief Class representing a shape that may be partially or totally dynamic.
///
///
/// A PartialShape may have:
///
/// \li Dynamic rank. (Informal notation: `?`)
/// \li Static rank, but dynamic dimensions on some or all axes.
///     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
/// \li Static rank, and static dimensions on all axes.
///     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
/// \ingroup ov_model_cpp_api
class OPENVINO_API PartialShape {
    using Dimensions = std::vector<Dimension>;

public:
    using value_type = Dimensions::value_type;
    using iterator = Dimensions::iterator;
    using const_iterator = Dimensions::const_iterator;
    using reverse_iterator = Dimensions::reverse_iterator;
    using const_reverse_iterator = Dimensions::const_reverse_iterator;

    /// \brief Constructs a shape with static rank from an initializer list of Dimension.
    /// \param init The Dimension values for the constructed shape.
    ///
    /// Examples:
    ///
    /// \code{.cpp}
    /// PartialShape s{2,3,4};                     // rank=3, all dimensions static
    /// PartialShape s{};                          // rank=0
    /// PartialShape s{2,Dimension::dynamic(),3};  // rank=3, dimension 1 dynamic
    /// \endcode
    PartialShape(std::initializer_list<Dimension> init);

    /// \brief Constructs a PartialShape with static rank from a vector of Dimension.
    /// \param dimensions The Dimension values for the constructed shape.
    PartialShape(std::vector<Dimension> dimensions);

    /// \brief Constructs a PartialShape with static rank from a vector of dimensions values.
    /// \param dimensions The Dimension values for the constructed shape.
    PartialShape(const std::vector<Dimension::value_type>& dimensions);

    /// \brief Constructs a static PartialShape with zero rank (the shape of a scalar).
    PartialShape();

    /// \brief Constructs a static PartialShape from a PartialShape.
    /// \param shape The PartialShape to convert into PartialShape.
    PartialShape(const Shape& shape);

    /// \brief Constructs a static PartialShape from a string.
    /// \param shape The string to parse into PartialShape.
    PartialShape(const std::string& shape);

    /// \brief Check if this shape is static.
    /// \return `true` if this shape is static, else `false`.
    ///
    /// A shape is considered static if it has static rank, and all dimensions of the shape
    /// are static.
    bool is_static() const;

    /// \brief Check if this shape is dynamic.
    /// \return `false` if this shape is static, else `true`.
    ///
    /// A shape is considered static if it has static rank, and all dimensions of the shape
    /// are static.
    bool is_dynamic() const {
        return !is_static();
    }
    /// \brief Get the rank of the shape.
    /// \return The rank of the shape. This will be Rank::dynamic() if the rank of
    ///         the shape is dynamic.
    Rank rank() const {
        return m_rank_is_static ? Rank(m_dimensions.size()) : Rank::dynamic();
    }
    /// \brief Construct a PartialShape with the given rank and all dimensions (if any) dynamic.
    /// \return A PartialShape with the given rank, and all dimensions (if any) dynamic.
    static PartialShape dynamic(Rank r = Rank::dynamic());
    /// \brief Check whether this shape is compatible with the argument, i.e., whether it is
    ///        possible to merge them.
    /// \param s The shape to be checked for compatibility with this shape.
    /// \return `true` if this shape is compatible with `s`, else `false`.
    ///
    /// Two shapes are compatible if
    /// \li one or both of them has dynamic rank, or
    /// \li both shapes have dynamic and equal rank, and their dimensions are elementwise
    ///     compatible (see Dimension::compatible()).
    bool compatible(const PartialShape& s) const;

    /// \brief Check whether this shape represents the same scheme as the argument.
    /// \param s The shape whose scheme is being compared with this shape.
    /// \return `true` if this shape represents the same scheme as `s`, else `false`.
    ///
    /// Two shapes `s1` and `s2` represent the same scheme if
    /// \li they both have dynamic rank, or
    /// \li they both have static and equal rank `r`, and for every `i` from `0` to `r-1`,
    ///     `s1[i]` represents the same scheme as `s2[i]` (see Dimension::same_scheme()).
    bool same_scheme(const PartialShape& s) const;

    /// \brief Check whether this shape is a relaxation of the argument.
    /// \param s The shape which is being compared against this shape.
    /// \return `true` if this shape relaxes `s`, else `false`.
    ///
    /// Intuitively, a PartialShape `s1` is said to _relax_ `s2` (or _is a
    /// relaxation_ of `s2`) if it is "more permissive" than `s2`. In other
    /// words, `s1` is a relaxation of `s2` if anything you can form by
    /// plugging things into the dynamic dimensions of `s2` is also
    /// something you can form by plugging things into the dynamic
    /// dimensions of `s1`, but not necessarily the other way around.
    ///
    /// `s1.relaxes(s2)` is equivalent to `s2.refines(s1)`.
    ///
    /// Formally, PartialShape `s1` is said to _relax_ PartialShape `s2`
    /// if:
    /// \li For every `i` from `0` to `r-1`,
    ///      either `s1[i]` contains s2[i].
    bool relaxes(const PartialShape& s) const;

    /// \brief Check whether this shape is a refinement of the argument.
    /// \param s The shape which is being compared against this shape.
    /// \return `true` if this shape refines `s`, else `false`.
    ///
    /// Intuitively, a PartialShape `s1` is said to _relax_ `s2` (or _is a
    /// relaxation_ of `s2`) if it is "less permissive" than `s2`. In other
    /// words, `s1` is a relaxation of `s2` if anything you can form by
    /// plugging things into the dynamic dimensions of `s1` is also
    /// something you can form by plugging things into the dynamic
    /// dimensions of `s2`, but not necessarily the other way around.
    ///
    /// `s1.refines(s2)` is equivalent to `s2.relaxes(s1)`.
    ///
    /// Formally, PartialShape `s1` is said to _refine_ PartialShape `s2`
    /// if:
    /// \li `s2` has dynamic rank, or
    /// \li `s1` and `s2` both have static rank `r`, and for every `i` from `0` to `r-1`,
    ///      either `s2[i]` is dynamic, or `s1[i]` == `s2[i]`.
    bool refines(const PartialShape& s) const;

    /// \brief Checks that this shape's rank is compatible with `r`, and, if this shape's
    ///        rank is dynamic and `r` is static, updates this shape to have a rank of `r`
    ///        with dimensions all dynamic.
    /// \return `true` if this shape's rank is compatible with `r`, else `false`.
    bool merge_rank(const Rank& r);

    /// \brief Convert a static PartialShape to a PartialShape.
    /// \return A new PartialShape `s` where `s[i] = size_t((*this)[i])`.
    /// \throws std::invalid_argument If this PartialShape is dynamic.
    Shape to_shape() const;

    /// \brief Returns `true` if all static dimensions of the tensor are non-negative, else
    ///        `false`.
    bool all_non_negative() const;

    /// \brief Index operator for PartialShape, with bound checking.
    /// \param i The index of the dimension being selected in range [-rank, rank).
    /// \return A reference to the `i`th Dimension of this shape.
    Dimension& operator[](std::ptrdiff_t i);
    /// \brief Index operator for PartialShape, with bound checking.
    /// \param i The index of the dimension being selected in range [-rank, rank).
    /// \return A reference to the `i`th Dimension of this shape.
    const Dimension& operator[](std::ptrdiff_t i) const;

    /// \brief Returns a vector of the dimensions. This has no meaning if dynamic.
    explicit operator std::vector<Dimension>() const {
        return m_dimensions;
    }
    friend OPENVINO_API std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
    friend OPENVINO_API PartialShape operator+(const PartialShape& s1, const PartialShape& s2);
    bool operator==(const PartialShape& partial_shape) const;
    bool operator!=(const PartialShape& partial_shape) const;
    /// Get the max bounding shape
    Shape get_max_shape() const;
    /// Get the min bounding shape
    Shape get_min_shape() const;
    /// Get the unique shape
    Shape get_shape() const;

    /// \brief Try to merge one shape into another.
    /// \param[in,out] dst The shape that `src` will be merged into.
    /// \param src The shape that will be merged into `dst`.
    /// \return `true` if merging succeeds, else `false`.
    ///
    /// Merges `src` into `dst`, returning `true` on success and `false` on failure. If
    /// `false` is returned, the effect on `dst` is unspecified.
    ///
    /// To merge two partial shapes `s1` and `s2` is to find the most permissive partial shape
    /// `s` that is no more permissive than `s1` or `s2`, if `s` exists. For example:
    ///
    /// \code
    ///        merge(?,?) -> ?
    ///        merge(?,{?,?}) -> {?,?}
    ///        merge({?,?},{?,?}) -> {?,?}
    ///        merge({1,2,3,4},?) -> {1,2,3,4}
    ///        merge({1,2},{1,?}) -> {1,2}
    ///        merge({1,2,?,?},{1,?,3,?}) -> {1,2,3,?}
    ///        merge({1,2,3},{1,2,3}) -> {1,2,3}
    ///
    ///        merge({1,?},{2,?}) fails [dimension 0 constraints are inconsistent]
    ///        merge({?,?},{?,?,?}) fails [ranks are inconsistent]
    /// \endcode
    ///
    /// This function (merge_into) performs the "merge" operation described above on `dst` and
    /// `src`, but overwrites `dst` with the result and returns `true` if merging is
    /// successful; if merging is unsuccessful, the function returns `false` and may make
    /// unspecified changes to `dst`.
    static bool merge_into(PartialShape& dst, const PartialShape& src);

    /// \brief Try to merge one shape into another along with implicit broadcasting
    static bool broadcast_merge_into(PartialShape& dst,
                                     const PartialShape& src,
                                     const ov::op::AutoBroadcastSpec& autob);

    /// \brief Returns a read/write iterator that points to the first
    ///        element in the shape. Iteration is done in ordinary
    ///        element order.
    iterator begin() noexcept {
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
        return m_dimensions.begin();
    }
    /// \brief Returns a read-only (constant) iterator that points to the
    ///        first element in the shape. Iteration is done in ordinary
    ///        element order.
    const_iterator begin() const noexcept {
        return cbegin();
    }
    /// \brief Returns a read/write iterator that points one past the last
    ///        element in the shape. Iteration is done in ordinary
    ///        element order.
    iterator end() noexcept {
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
        return m_dimensions.end();
    }
    /// \brief Returns a read-only (constant) iterator that points one past
    ///        the last element in the shape. Iteration is done in ordinary
    ///        element order.
    const_iterator end() const noexcept {
        return cend();
    }
    /// \brief Returns a read/write reverse iterator that points to the
    ///        last element in the shape. Iteration is done in reverse
    ///        element order.
    reverse_iterator rbegin() noexcept {
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
        return m_dimensions.rbegin();
    }
    /// \brief Returns a read-only (constant) reverse iterator that points
    ///        to the last element in the shape. Iteration is done in
    ///        reverse element order.
    const_reverse_iterator rbegin() const noexcept {
        return crbegin();
    }
    /// \brief Returns a read/write reverse iterator that points to one
    ///        before the first element in the shape. Iteration is done
    ///        in reverse element order.
    reverse_iterator rend() noexcept {
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
        return m_dimensions.rend();
    }
    /// \brief Returns a read-only (constant) reverse iterator that points
    ///        to one before the first element in the shape. Iteration
    ///        is done in reverse element order.
    const_reverse_iterator rend() const noexcept {
        return crend();
    }
    /// \brief Returns a read-only (constant) iterator that points to the
    ///        first element in the shape. Iteration is done in ordinary
    ///        element order.
    const_iterator cbegin() const noexcept {
        return m_dimensions.cbegin();
    }
    /// \brief Returns a read-only (constant) iterator that points one past
    ///        the last element in the shape. Iteration is done in ordinary
    ///        element order.
    const_iterator cend() const noexcept {
        return m_dimensions.cend();
    }
    /// \brief Returns a read-only (constant) reverse iterator that points
    ///        to the last element in the shape. Iteration is done in
    ///        reverse element order.
    const_reverse_iterator crbegin() const noexcept {
        return m_dimensions.crbegin();
    }
    /// \brief Returns a read-only (constant) reverse iterator that points
    ///        to one before the first element in the shape. Iteration
    ///        is done in reverse element order.
    const_reverse_iterator crend() const noexcept {
        return m_dimensions.crend();
    }

    /// \brief Resizes dimensions container to contain count elements
    void resize(size_t count) {
        m_dimensions.resize(count);
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
    }
    /// \brief Returns size of dimension vector. Requires rank to be static
    size_t size() const {
        OPENVINO_ASSERT(rank().is_static());
        return m_dimensions.size();
    }
    /// \brief Returns a read/write iterator that points to the inserted element in the shape.
    iterator insert(iterator position, const Dimension& val) {
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
        return m_dimensions.insert(position, val);
    }
    /// \brief Inserts count copies of the value before position
    void insert(iterator position, size_t n, const Dimension& val) {
        m_dimensions.insert(position, n, val);
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
    }
    /// \brief Inserts elements from range [first, last) before position
    template <class InputIterator>
    void insert(iterator position, InputIterator first, InputIterator last) {
        m_dimensions.insert(position, first, last);
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
    }
    /// \brief Requests that the dimensions vector capacity be enough to contain n elements
    void reserve(size_t n) {
        m_dimensions.reserve(n);
    }
    /// \brief push element to the end of partial shape
    void push_back(const Dimension& val) {
        m_dimensions.push_back(val);
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
    }
    /// \brief emplace element to the end of partial shape
    template <class... Args>
    void emplace_back(Args&&... args) {
        m_dimensions.emplace_back(std::forward<Args>(args)...);
        m_rank_is_static = true;
        m_shape_type = ShapeType::SHAPE_IS_UPDATED;
    }

    /// \brief String representation of PartialShape
    std::string to_string() const;

private:
    // Private constructor for PartialShape::dynamic().
    PartialShape(bool rank_is_static, std::vector<Dimension> dimensions);

    // True if the shape's rank is static.
    bool m_rank_is_static;

    /// \brief PartialShape types. The shape type is lazily evaluated by calling the is_static()
    /// method.
    ///
    /// \details It is highly recommended to avoid using the Dimension& operator[](size_t)
    /// operator. It sets the shape type to SHAPE_IS_UPDATED and disables shape type caching.
    /// Thus, the is_static method will have linear complexity because the shape is not
    /// guaranteed to remain static or dynamic.
    mutable enum class ShapeType {
        SHAPE_IS_UNKNOWN,  // The shape type is unknown and should be calculated by checking all
                           // dimensions.
        SHAPE_IS_UPDATED,  // User has retained a link to one dimension. Therefore, we can't
                           // guarantee that the shape will remain static or dynamic, and its
                           // type will always be evaluated.
        SHAPE_IS_STATIC,   // The shape type is known and static. Also there are no any retained
                           // dimensions by non-constant reference.
        SHAPE_IS_DYNAMIC   // The shape type is dynamic and there are no any retained dimensions
                           // by non-constant reference.
    } m_shape_type{ShapeType::SHAPE_IS_UNKNOWN};

    // PartialShape dimensions. This has no meaning if m_rank_is_static is false.
    Dimensions m_dimensions;
};

/// \brief Elementwise addition of two PartialShape objects.
/// \param s1 Left operand for addition.
/// \param s2 Right operand for addition.
/// \return The result of elementwise adding `s1` to `s2` (see description).
/// \throws std::invalid_argument If `s1` and `s2` have inconsistent ranks.
///
/// \li If `s1` or `s2` has dynamic rank, returns PartialShape::dynamic().
/// \li If `s1 and `s2` both have static rank, and their ranks are unequal, throws
///     std::invalid_argument.
/// \li If `s1` and `s2` both have static rank, and their ranks are equal,
///     returns a new shape whose `i`th dimension is `s1[i] + s2[i]`.
OPENVINO_API PartialShape operator+(const PartialShape& s1, const PartialShape& s2);

/// \brief Inserts a human-readable representation of a PartialShape into an output stream.
/// \param str The output stream targeted for insertion.
/// \param shape The shape to be inserted into `str`.
/// \return A reference to `str` after insertion.
///
/// The output to the stream is in "informal" notation. In other words:
///
/// \li If `shape` has dynamic rank, inserts the string `?`.
/// \li If `shape` has static rank, inserts the string `{`, then inserts each dimension
///     of `shape` into the output stream separated by commas, then inserts `}`.
///
/// Example:
///
/// \code{.cpp}
/// PartialShape s1{PartialShape::dynamic())};
/// PartialShape s2{};
/// PartialShape s3{1,Dimension::dynamic(),2,3};
/// PartialShape s4{2,3,4};
/// std::cout << s1 << std::endl
///           << s2 << std::endl
///           << s3 << std::endl
///           << s4 << std::endl;
/// \endcode
///
/// Output:
///
/// \code
/// ?
/// {}
/// {1,?,2,3}
/// {2,3,4}
/// \endcode
OPENVINO_API
std::ostream& operator<<(std::ostream& str, const PartialShape& shape);

template <>
class OPENVINO_API AttributeAdapter<ov::PartialShape> : public DirectValueAccessor<ov::PartialShape> {
public:
    AttributeAdapter(ov::PartialShape& value) : DirectValueAccessor<ov::PartialShape>(value) {}

    OPENVINO_RTTI("AttributeAdapter<PartialShape>");
};
}  // namespace ov
