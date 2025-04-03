/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef GRAPH_UTILS_JSON_HPP
#define GRAPH_UTILS_JSON_HPP

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace json {

/*!
 * \brief template to select type based on condition
 * For example, if_else_type<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam Else typename to be returned if cond is false
 */

template <bool cond, typename Then, typename Else>
struct if_else_type;

/*!
 * \brief generic serialization json
 * \tparam T the type to be serialized
 */
template <typename T>
struct json_handler;

template <typename T>
struct common_json;

/*!
 * \brief json to write any type.
 */
class json_writer_t {
public:
    json_writer_t(std::ostream *os) : os_(os) {}
    /*!
     * \brief object begin
     * \param multi_line whether to start an multi_line array.
     */
    inline void begin_object();
    /*! \brief object end. */
    inline void end_object();
    /*!
     * \brief write key value pair in the object.
     */
    template <typename valuetype>
    inline void write_keyvalue(const std::string &key, const valuetype &value);
    /*!
     * \brief write a number.
     */
    template <typename valuetype>
    inline void write_number(const valuetype &v);
    /*!
     * \brief write a string.
     */
    inline void write_string(const std::string &s);
    /*!
     * \brief array begin.
     * \param multi_line if true, write multi_line.
     */
    inline void begin_array(bool multi_line = true);
    /*! \brief array end. */
    inline void end_array();
    /*!
     * \brief write array separator.
     */
    inline void write_array_seperator();
    /*!
     * \brief write array value.
     */
    template <typename valuetype>
    inline void write_array_item(const valuetype &value);

private:
    std::ostream *os_;
    /*!
     * \brief record how many element in the current scope.
     */
    std::vector<size_t> scope_count_;
    /*! \brief record if current pos is a multiline scope */
    std::vector<bool> scope_multi_line_;
    /*!
     * \brief write seperator space and newlines
     */
    inline void write_seperator();
};

class json_reader_t {
public:
    explicit json_reader_t(std::istream *is) : is_(is) {}
    /*!
     * \brief parse json string.
     */
    inline void read_string(std::string *out_str);
    /*!
     * \brief read number.
     */
    template <typename valuetype>
    inline void read_number(valuetype *out_value);
    /*!
     * \brief parse an object begin.
     */
    inline void begin_object();
    /*!
     * \brief parse an object end.
     */
    inline void begin_array();
    /*!
     * \brief read next object, if true, will read next object.
     */
    inline bool next_object_item(std::string *out_key);
    /*!
     * \brief read next object, if true, will read next object.
     */
    inline bool next_array_item();
    /*!
     * \brief read next value.
     */
    template <typename valuetype>
    inline void read(valuetype *out_value);

private:
    std::istream *is_;
    /*!
     * \brief record element size in the current.
     */
    std::vector<size_t> scope_count_;
    /*!
     * \brief read next nonspace char.
     */
    inline int next_nonspace();
    inline int peeknext_nonspace();
    /*!
   * \brief get the next char from the input.
   */
    inline int next_char();
    inline int peeknext_char();
};

class read_helper_t {
public:
    /*!
   * \brief declare field
   */
    template <typename T>
    inline void declare_field(const std::string &key, T *addr) {
        //declare_fieldInternal(key, addr);
        if (map_.count(key) == 0) {
            entry_t e;
            e.func = reader_function<T>;
            e.addr = static_cast<void *>(addr);
            map_[key] = e;
        }
    }
    /*!
   * \brief read all fields according to declare.
   */
    inline bool read_fields(json_reader_t *reader);

private:
    /*!
     * \brief reader function to store T.
     */
    template <typename T>
    inline static void reader_function(json_reader_t *reader, void *addr);
    /*! \brief callback type to reader function */
    typedef void (*readfunc)(json_reader_t *reader, void *addr);
    /*! \brief data entry */
    struct entry_t {
        readfunc func;
        /*! \brief store the address data for reading json*/
        void *addr;
    };
    /*! \brief reader callback */
    std::map<std::string, entry_t> map_;
};

template <typename then, typename other>
struct if_else_type<true, then, other> {
    using type = then;
};

template <typename then, typename other>
struct if_else_type<false, then, other> {
    using type = other;
};

template <typename valuetype>
struct num_json_t {
    inline static void write(json_writer_t *writer, const valuetype &value) {
        writer->write_number<valuetype>(value);
    }
    inline static void read(json_reader_t *reader, valuetype *value) {
        reader->read_number<valuetype>(value);
    }
};

template <typename valuetype>
struct common_json {
    inline static void write(json_writer_t *writer, const valuetype &value) {
        value.save(writer);
    }
    inline static void read(json_reader_t *reader, valuetype *value) {
        value->load(reader);
    }
};

template <typename valuetype>
struct common_json<std::shared_ptr<valuetype>> {
    inline static void write(
            json_writer_t *writer, const std::shared_ptr<valuetype> &value) {
        auto *v = value.get();
        v->save(writer);
    }

    inline static void read(
            json_reader_t *reader, std::shared_ptr<valuetype> *value) {
        auto ptr = std::make_shared<valuetype>();
        auto *v = ptr.get();
        v->load(reader);
        *value = std::move(ptr);
    }
};

template <typename CT>
struct array_json_t {
    inline static void write(json_writer_t *writer, const CT &array) {
        writer->begin_array();
        for (typename CT::const_iterator it = array.begin(); it != array.end();
                ++it) {
            writer->write_array_item(*it);
        }
        writer->end_array();
    }
    inline static void read(json_reader_t *reader, CT *array) {
        using elemtype = typename CT::value_type;
        array->clear();
        reader->begin_array();
        while (reader->next_array_item()) {
            elemtype value;
            json_handler<elemtype>::read(reader, &value);
            array->insert(array->end(), value);
        }
    }
};

template <typename CT>
struct map_json_t {
    inline static void write(json_writer_t *writer, const CT &map) {
        writer->begin_object();
        for (typename CT::const_iterator it = map.begin(); it != map.end();
                ++it) {
            writer->write_keyvalue(it->first, it->second);
        }
        writer->end_object();
    }
    inline static void read(json_reader_t *reader, CT *map) {
        using elemtype = typename CT::mapped_type;
        map->clear();
        reader->begin_object();
        std::string key;
        while (reader->next_object_item(&key)) {
            elemtype value;
            reader->read(&value);
            (*map)[key] = std::move(value);
        }
    }
};

template <>
struct json_handler<std::string> {
    inline static void write(json_writer_t *writer, const std::string &value) {
        writer->write_string(value);
    }
    inline static void read(json_reader_t *reader, std::string *str) {
        reader->read_string(str);
    }
};

template <typename T>
struct json_handler<std::map<std::string, T>>
    : public map_json_t<std::map<std::string, T>> {};

template <typename T>
struct json_handler<std::unordered_map<std::string, T>>
    : public map_json_t<std::unordered_map<std::string, T>> {};

template <typename T>
struct json_handler<std::vector<T>> : public array_json_t<std::vector<T>> {};

template <typename T>
struct json_handler<std::list<T>> : public array_json_t<std::list<T>> {};
/*!
 * \brief generic serialization json
 */
template <typename T>
struct json_handler {
    inline static void write(json_writer_t *writer, const T &data) {
        using Tjson = typename if_else_type<std::is_arithmetic<T>::value,
                num_json_t<T>, common_json<T>>::type;
        Tjson::write(writer, data);
    }
    inline static void read(json_reader_t *reader, T *data) {
        using Tjson = typename if_else_type<std::is_arithmetic<T>::value,
                num_json_t<T>, common_json<T>>::type;
        Tjson::read(reader, data);
    }
};

inline void json_writer_t::begin_object() {
    *os_ << "{";
    scope_multi_line_.push_back(true);
    scope_count_.push_back(0);
}

template <typename valuetype>
inline void json_writer_t::write_keyvalue(
        const std::string &key, const valuetype &value) {
    if (scope_count_.back() > 0) { *os_ << ","; }
    write_seperator();
    *os_ << '\"';
    *os_ << key;
    *os_ << "\": ";
    scope_count_.back() += 1;
    json_handler<valuetype>::write(this, value);
}

template <typename valuetype>
inline void json_writer_t::write_number(const valuetype &v) {
    *os_ << v;
}

inline void json_writer_t::write_string(const std::string &s) {
    *os_ << '\"';
    for (size_t i = 0; i < s.length(); ++i) {
        char ch = s[i];
        switch (ch) {
            case '\r': *os_ << "\\r"; break;
            case '\n': *os_ << "\\n"; break;
            case '\\': *os_ << "\\\\"; break;
            case '\t': *os_ << "\\t"; break;
            case '\"': *os_ << "\\\""; break;
            default: *os_ << ch;
        }
    }
    *os_ << '\"';
}

inline void json_writer_t::begin_array(bool multi_line) {
    *os_ << '[';
    scope_multi_line_.push_back(multi_line);
    scope_count_.push_back(0);
}

inline void json_writer_t::end_array() {
    if (!scope_count_.empty() && !scope_multi_line_.empty()) {
        bool newline = scope_multi_line_.back();
        size_t nelem = scope_count_.back();
        scope_multi_line_.pop_back();
        scope_count_.pop_back();
        if (newline && nelem != 0) write_seperator();
    }
    *os_ << ']';
}

inline void json_writer_t::write_array_seperator() {
    if (scope_count_.back() != 0) { *os_ << ", "; }
    scope_count_.back() += 1;
    write_seperator();
}

template <typename valuetype>
inline void json_writer_t::write_array_item(const valuetype &value) {
    this->write_array_seperator();
    json::json_handler<valuetype>::write(this, value);
}

inline void json_writer_t::end_object() {
    if (!scope_count_.empty() && !scope_multi_line_.empty()) {
        bool newline = scope_multi_line_.back();
        size_t nelem = scope_count_.back();
        scope_multi_line_.pop_back();
        scope_count_.pop_back();
        if (newline && nelem != 0) write_seperator();
    }
    *os_ << '}';
}

inline void json_writer_t::write_seperator() {
    if (scope_multi_line_.empty() || scope_multi_line_.back()) {
        *os_ << '\n';
        *os_ << std::string(scope_multi_line_.size() * 2, ' ');
    }
}

inline int json_reader_t::next_char() {
    return is_->get();
}

inline int json_reader_t::peeknext_char() {
    return is_->peek();
}

inline int json_reader_t::next_nonspace() {
    int ch;
    do {
        ch = next_char();
    } while (isspace(ch));
    return ch;
}

inline int json_reader_t::peeknext_nonspace() {
    int ch;
    while (true) {
        ch = peeknext_char();
        if (!isspace(ch)) break;
        next_char();
    }
    return ch;
}

inline void json_reader_t::read_string(std::string *out_str) {
    int ch = next_nonspace();
    if (ch == '\"') {
        std::ostringstream output;
        while (true) {
            ch = next_char();
            if (ch == '\\') {
                char sch = static_cast<char>(next_char());
                switch (sch) {
                    case 'r': output << "\r"; break;
                    case 'n': output << "\r"; break;
                    case '\\': output << "\r"; break;
                    case 't': output << "\r"; break;
                    case '\"': output << "\r"; break;
                    default: throw("unknown string escape.");
                }
            } else {
                if (ch == '\"') break;
                output << static_cast<char>(ch);
            }
            if (ch == EOF || ch == '\r' || ch == '\n') {
                throw("error at!");
                return;
            }
        }
        *out_str = output.str();
    }
}

template <typename valuetype>
inline void json_reader_t::read_number(valuetype *out_value) {
    *is_ >> *out_value;
}

inline void json_reader_t::begin_object() {
    int ch = next_nonspace();
    if (ch == '{') { scope_count_.push_back(0); }
}

inline void json_reader_t::begin_array() {
    int ch = next_nonspace();
    if (ch == '[') { scope_count_.push_back(0); }
}

inline bool json_reader_t::next_object_item(std::string *out_key) {
    bool next = true;
    if (scope_count_.empty()) { return false; }
    if (scope_count_.back() != 0) {
        int ch = next_nonspace();
        if (ch == EOF) {
            next = false;
        } else if (ch == '}') {
            next = false;
        } else {
            if (ch != ',') { return false; }
        }
    } else {
        int ch = peeknext_nonspace();
        if (ch == '}') {
            next_char();
            next = false;
        }
    }
    if (!next) {
        scope_count_.pop_back();
        return false;
    } else {
        scope_count_.back() += 1;
        read_string(out_key);
        int ch = next_nonspace();
        return (ch == ':');
    }
}

inline bool json_reader_t::next_array_item() {
    bool next = true;
    if (scope_count_.empty()) { return false; }
    if (scope_count_.back() != 0) {
        int ch = next_nonspace();
        if (ch == EOF) {
            next = false;
        } else if (ch == ']') {
            next = false;
        } else {
            if (ch != ',') { return false; }
        }
    } else {
        int ch = peeknext_nonspace();
        if (ch == ']') {
            next_char();
            next = false;
        }
    }
    if (!next) {
        scope_count_.pop_back();
        return false;
    } else {
        scope_count_.back() += 1;
        return true;
    }
}

template <typename valuetype>
inline void json_reader_t::read(valuetype *out_value) {
    json::json_handler<valuetype>::read(this, out_value);
}

inline bool read_helper_t::read_fields(json_reader_t *reader) {
    reader->begin_object();
    std::map<std::string, int> visited;
    std::string key;
    while (reader->next_object_item(&key)) {
        if (map_.count(key) != 0) {
            entry_t e = map_[key];
            (*e.func)(reader, e.addr);
            visited[key] = 0;
        }
    }
    if (visited.size() != map_.size()) {
        for (std::map<std::string, entry_t>::iterator it = map_.begin();
                it != map_.end(); ++it) {
            if (visited.count(it->first) != 1) { return false; }
        }
    }
    return true;
}

template <typename T>
inline void read_helper_t::reader_function(json_reader_t *reader, void *addr) {
    json::json_handler<T>::read(reader, static_cast<T *>(addr));
}

} // namespace json
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
