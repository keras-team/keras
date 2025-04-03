#!/usr/bin/python
#
# Copyright (C) 2009 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts Python data into data for Google Visualization API clients.

This library can be used to create a google.visualization.DataTable usable by
visualizations built on the Google Visualization API. Output formats are raw
JSON, JSON response, JavaScript, CSV, and HTML table.

See http://code.google.com/apis/visualization/ for documentation on the
Google Visualization API.
"""

__author__ = "Amit Weinstein, Misha Seltzer, Jacob Baskin"

import csv
import datetime
import decimal
try:
  import html  # Python version 3.2 or higher
except ImportError:
  import cgi as html  # Only used for .escape()
import numbers
import json
import types

import six


class DataTableException(Exception):
  """The general exception object thrown by DataTable."""
  pass


class DataTableJSONEncoder(json.JSONEncoder):
  """JSON encoder that handles date/time/datetime objects correctly."""

  def __init__(self):
    json.JSONEncoder.__init__(self,
                              separators=(",", ":"),
                              ensure_ascii=False)

  def default(self, o):
    if isinstance(o, datetime.datetime):
      if o.microsecond == 0:
        # If the time doesn't have ms-resolution, leave it out to keep
        # things smaller.
        return "Date(%d,%d,%d,%d,%d,%d)" % (
            o.year, o.month - 1, o.day, o.hour, o.minute, o.second)
      else:
        return "Date(%d,%d,%d,%d,%d,%d,%d)" % (
            o.year, o.month - 1, o.day, o.hour, o.minute, o.second,
            o.microsecond / 1000)
    elif isinstance(o, datetime.date):
      return "Date(%d,%d,%d)" % (o.year, o.month - 1, o.day)
    elif isinstance(o, datetime.time):
      return [o.hour, o.minute, o.second]
    else:
      return super(DataTableJSONEncoder, self).default(o)


class DataTable(object):
  """Wraps the data to convert to a Google Visualization API DataTable.

  Create this object, populate it with data, then call one of the ToJS...
  methods to return a string representation of the data in the format described.

  You can clear all data from the object to reuse it, but you cannot clear
  individual cells, rows, or columns. You also cannot modify the table schema
  specified in the class constructor.

  You can add new data one or more rows at a time. All data added to an
  instantiated DataTable must conform to the schema passed in to __init__().

  You can reorder the columns in the output table, and also specify row sorting
  order by column. The default column order is according to the original
  table_description parameter. Default row sort order is ascending, by column
  1 values. For a dictionary, we sort the keys for order.

  The data and the table_description are closely tied, as described here:

  The table schema is defined in the class constructor's table_description
  parameter. The user defines each column using a tuple of
  (id[, type[, label[, custom_properties]]]). The default value for type is
  string, label is the same as ID if not specified, and custom properties is
  an empty dictionary if not specified.

  table_description is a dictionary or list, containing one or more column
  descriptor tuples, nested dictionaries, and lists. Each dictionary key, list
  element, or dictionary element must eventually be defined as
  a column description tuple. Here's an example of a dictionary where the key
  is a tuple, and the value is a list of two tuples:
    {('a', 'number'): [('b', 'number'), ('c', 'string')]}

  This flexibility in data entry enables you to build and manipulate your data
  in a Python structure that makes sense for your program.

  Add data to the table using the same nested design as the table's
  table_description, replacing column descriptor tuples with cell data, and
  each row is an element in the top level collection. This will be a bit
  clearer after you look at the following examples showing the
  table_description, matching data, and the resulting table:

  Columns as list of tuples [col1, col2, col3]
    table_description: [('a', 'number'), ('b', 'string')]
    AppendData( [[1, 'z'], [2, 'w'], [4, 'o'], [5, 'k']] )
    Table:
    a  b   <--- these are column ids/labels
    1  z
    2  w
    4  o
    5  k

  Dictionary of columns, where key is a column, and value is a list of
  columns  {col1: [col2, col3]}
    table_description: {('a', 'number'): [('b', 'number'), ('c', 'string')]}
    AppendData( data: {1: [2, 'z'], 3: [4, 'w']}
    Table:
    a  b  c
    1  2  z
    3  4  w

  Dictionary where key is a column, and the value is itself a dictionary of
  columns {col1: {col2, col3}}
    table_description: {('a', 'number'): {'b': 'number', 'c': 'string'}}
    AppendData( data: {1: {'b': 2, 'c': 'z'}, 3: {'b': 4, 'c': 'w'}}
    Table:
    a  b  c
    1  2  z
    3  4  w
  """

  def __init__(self, table_description, data=None, custom_properties=None):
    """Initialize the data table from a table schema and (optionally) data.

    See the class documentation for more information on table schema and data
    values.

    Args:
      table_description: A table schema, following one of the formats described
                         in TableDescriptionParser(). Schemas describe the
                         column names, data types, and labels. See
                         TableDescriptionParser() for acceptable formats.
      data: Optional. If given, fills the table with the given data. The data
            structure must be consistent with schema in table_description. See
            the class documentation for more information on acceptable data. You
            can add data later by calling AppendData().
      custom_properties: Optional. A dictionary from string to string that
                         goes into the table's custom properties. This can be
                         later changed by changing self.custom_properties.

    Raises:
      DataTableException: Raised if the data and the description did not match,
                          or did not use the supported formats.
    """
    self.__columns = self.TableDescriptionParser(table_description)
    self.__data = []
    self.custom_properties = {}
    if custom_properties is not None:
      self.custom_properties = custom_properties
    if data:
      self.LoadData(data)

  @staticmethod
  def CoerceValue(value, value_type):
    """Coerces a single value into the type expected for its column.

    Internal helper method.

    Args:
      value: The value which should be converted
      value_type: One of "string", "number", "boolean", "date", "datetime" or
                  "timeofday".

    Returns:
      An item of the Python type appropriate to the given value_type. Strings
      are also converted to Unicode using UTF-8 encoding if necessary.
      If a tuple is given, it should be in one of the following forms:
        - (value, formatted value)
        - (value, formatted value, custom properties)
      where the formatted value is a string, and custom properties is a
      dictionary of the custom properties for this cell.
      To specify custom properties without specifying formatted value, one can
      pass None as the formatted value.
      One can also have a null-valued cell with formatted value and/or custom
      properties by specifying None for the value.
      This method ignores the custom properties except for checking that it is a
      dictionary. The custom properties are handled in the ToJSon and ToJSCode
      methods.
      The real type of the given value is not strictly checked. For example,
      any type can be used for string - as we simply take its str( ) and for
      boolean value we just check "if value".
      Examples:
        CoerceValue(None, "string") returns None
        CoerceValue((5, "5$"), "number") returns (5, "5$")
        CoerceValue(100, "string") returns "100"
        CoerceValue(0, "boolean") returns False

    Raises:
      DataTableException: The value and type did not match in a not-recoverable
                          way, for example given value 'abc' for type 'number'.
    """
    if isinstance(value, tuple):
      # In case of a tuple, we run the same function on the value itself and
      # add the formatted value.
      if (len(value) not in [2, 3] or
          (len(value) == 3 and not isinstance(value[2], dict))):
        raise DataTableException("Wrong format for value and formatting - %s." %
                                 str(value))
      if not isinstance(value[1], six.string_types + (type(None),)):
        raise DataTableException("Formatted value is not string, given %s." %
                                 type(value[1]))
      js_value = DataTable.CoerceValue(value[0], value_type)
      return (js_value,) + value[1:]

    t_value = type(value)
    if value is None:
      return value
    if value_type == "boolean":
      return bool(value)

    elif value_type == "number":
      if isinstance(value, numbers.Integral):
        return int(value)
      if isinstance(value, (numbers.Real, decimal.Decimal)):
        return float(value)
      raise DataTableException("Wrong type %s when expected number" % t_value)

    elif value_type == "string":
      if isinstance(value, six.text_type):
        return value
      if isinstance(value, bytes):
        return six.text_type(value, encoding="utf-8")
      else:
        return six.text_type(value)

    elif value_type == "date":
      if isinstance(value, datetime.datetime):
        return datetime.date(value.year, value.month, value.day)
      elif isinstance(value, datetime.date):
        return value
      else:
        raise DataTableException("Wrong type %s when expected date" % t_value)

    elif value_type == "timeofday":
      if isinstance(value, datetime.datetime):
        return datetime.time(value.hour, value.minute, value.second)
      elif isinstance(value, datetime.time):
        return value
      else:
        raise DataTableException("Wrong type %s when expected time" % t_value)

    elif value_type == "datetime":
      if isinstance(value, datetime.datetime):
        return value
      else:
        raise DataTableException("Wrong type %s when expected datetime" %
                                 t_value)
    # If we got here, it means the given value_type was not one of the
    # supported types.
    raise DataTableException("Unsupported type %s" % value_type)

  @staticmethod
  def EscapeForJSCode(encoder, value):
    if value is None:
      return "null"
    elif isinstance(value, datetime.datetime):
      if value.microsecond == 0:
        # If it's not ms-resolution, leave that out to save space.
        return "new Date(%d,%d,%d,%d,%d,%d)" % (value.year,
                                                value.month - 1,  # To match JS
                                                value.day,
                                                value.hour,
                                                value.minute,
                                                value.second)
      else:
        return "new Date(%d,%d,%d,%d,%d,%d,%d)" % (value.year,
                                                   value.month - 1,  # match JS
                                                   value.day,
                                                   value.hour,
                                                   value.minute,
                                                   value.second,
                                                   value.microsecond / 1000)
    elif isinstance(value, datetime.date):
      return "new Date(%d,%d,%d)" % (value.year, value.month - 1, value.day)
    else:
      return encoder.encode(value)

  @staticmethod
  def ToString(value):
    if value is None:
      return "(empty)"
    elif isinstance(value, (datetime.datetime,
                            datetime.date,
                            datetime.time)):
      return str(value)
    elif isinstance(value, six.text_type):
      return value
    elif isinstance(value, bool):
      return str(value).lower()
    elif isinstance(value, bytes):
      return six.text_type(value, encoding="utf-8")
    else:
      return six.text_type(value)

  @staticmethod
  def ColumnTypeParser(description):
    """Parses a single column description. Internal helper method.

    Args:
      description: a column description in the possible formats:
       'id'
       ('id',)
       ('id', 'type')
       ('id', 'type', 'label')
       ('id', 'type', 'label', {'custom_prop1': 'custom_val1'})
    Returns:
      Dictionary with the following keys: id, label, type, and
      custom_properties where:
        - If label not given, it equals the id.
        - If type not given, string is used by default.
        - If custom properties are not given, an empty dictionary is used by
          default.

    Raises:
      DataTableException: The column description did not match the RE, or
          unsupported type was passed.
    """
    if not description:
      raise DataTableException("Description error: empty description given")

    if not isinstance(description, (six.string_types, tuple)):
      raise DataTableException("Description error: expected either string or "
                               "tuple, got %s." % type(description))

    if isinstance(description, six.string_types):
      description = (description,)

    # According to the tuple's length, we fill the keys
    # We verify everything is of type string
    for elem in description[:3]:
      if not isinstance(elem, six.string_types):
        raise DataTableException("Description error: expected tuple of "
                                 "strings, current element of type %s." %
                                 type(elem))
    desc_dict = {"id": description[0],
                 "label": description[0],
                 "type": "string",
                 "custom_properties": {}}
    if len(description) > 1:
      desc_dict["type"] = description[1].lower()
      if len(description) > 2:
        desc_dict["label"] = description[2]
        if len(description) > 3:
          if not isinstance(description[3], dict):
            raise DataTableException("Description error: expected custom "
                                     "properties of type dict, current element "
                                     "of type %s." % type(description[3]))
          desc_dict["custom_properties"] = description[3]
          if len(description) > 4:
            raise DataTableException("Description error: tuple of length > 4")
    if desc_dict["type"] not in ["string", "number", "boolean",
                                 "date", "datetime", "timeofday"]:
      raise DataTableException(
          "Description error: unsupported type '%s'" % desc_dict["type"])
    return desc_dict

  @staticmethod
  def TableDescriptionParser(table_description, depth=0):
    """Parses the table_description object for internal use.

    Parses the user-submitted table description into an internal format used
    by the Python DataTable class. Returns the flat list of parsed columns.

    Args:
      table_description: A description of the table which should comply
                         with one of the formats described below.
      depth: Optional. The depth of the first level in the current description.
             Used by recursive calls to this function.

    Returns:
      List of columns, where each column represented by a dictionary with the
      keys: id, label, type, depth, container which means the following:
      - id: the id of the column
      - name: The name of the column
      - type: The datatype of the elements in this column. Allowed types are
              described in ColumnTypeParser().
      - depth: The depth of this column in the table description
      - container: 'dict', 'iter' or 'scalar' for parsing the format easily.
      - custom_properties: The custom properties for this column.
      The returned description is flattened regardless of how it was given.

    Raises:
      DataTableException: Error in a column description or in the description
                          structure.

    Examples:
      A column description can be of the following forms:
       'id'
       ('id',)
       ('id', 'type')
       ('id', 'type', 'label')
       ('id', 'type', 'label', {'custom_prop1': 'custom_val1'})
       or as a dictionary:
       'id': 'type'
       'id': ('type',)
       'id': ('type', 'label')
       'id': ('type', 'label', {'custom_prop1': 'custom_val1'})
      If the type is not specified, we treat it as string.
      If no specific label is given, the label is simply the id.
      If no custom properties are given, we use an empty dictionary.

      input: [('a', 'date'), ('b', 'timeofday', 'b', {'foo': 'bar'})]
      output: [{'id': 'a', 'label': 'a', 'type': 'date',
                'depth': 0, 'container': 'iter', 'custom_properties': {}},
               {'id': 'b', 'label': 'b', 'type': 'timeofday',
                'depth': 0, 'container': 'iter',
                'custom_properties': {'foo': 'bar'}}]

      input: {'a': [('b', 'number'), ('c', 'string', 'column c')]}
      output: [{'id': 'a', 'label': 'a', 'type': 'string',
                'depth': 0, 'container': 'dict', 'custom_properties': {}},
               {'id': 'b', 'label': 'b', 'type': 'number',
                'depth': 1, 'container': 'iter', 'custom_properties': {}},
               {'id': 'c', 'label': 'column c', 'type': 'string',
                'depth': 1, 'container': 'iter', 'custom_properties': {}}]

      input:  {('a', 'number', 'column a'): { 'b': 'number', 'c': 'string'}}
      output: [{'id': 'a', 'label': 'column a', 'type': 'number',
                'depth': 0, 'container': 'dict', 'custom_properties': {}},
               {'id': 'b', 'label': 'b', 'type': 'number',
                'depth': 1, 'container': 'dict', 'custom_properties': {}},
               {'id': 'c', 'label': 'c', 'type': 'string',
                'depth': 1, 'container': 'dict', 'custom_properties': {}}]

      input: { ('w', 'string', 'word'): ('c', 'number', 'count') }
      output: [{'id': 'w', 'label': 'word', 'type': 'string',
                'depth': 0, 'container': 'dict', 'custom_properties': {}},
               {'id': 'c', 'label': 'count', 'type': 'number',
                'depth': 1, 'container': 'scalar', 'custom_properties': {}}]

      input: {'a': ('number', 'column a'), 'b': ('string', 'column b')}
      output: [{'id': 'a', 'label': 'column a', 'type': 'number', 'depth': 0,
               'container': 'dict', 'custom_properties': {}},
               {'id': 'b', 'label': 'column b', 'type': 'string', 'depth': 0,
               'container': 'dict', 'custom_properties': {}}

      NOTE: there might be ambiguity in the case of a dictionary representation
      of a single column. For example, the following description can be parsed
      in 2 different ways: {'a': ('b', 'c')} can be thought of a single column
      with the id 'a', of type 'b' and the label 'c', or as 2 columns: one named
      'a', and the other named 'b' of type 'c'. We choose the first option by
      default, and in case the second option is the right one, it is possible to
      make the key into a tuple (i.e. {('a',): ('b', 'c')}) or add more info
      into the tuple, thus making it look like this: {'a': ('b', 'c', 'b', {})}
      -- second 'b' is the label, and {} is the custom properties field.
    """
    # For the recursion step, we check for a scalar object (string or tuple)
    if isinstance(table_description, (six.string_types, tuple)):
      parsed_col = DataTable.ColumnTypeParser(table_description)
      parsed_col["depth"] = depth
      parsed_col["container"] = "scalar"
      return [parsed_col]

    # Since it is not scalar, table_description must be iterable.
    if not hasattr(table_description, "__iter__"):
      raise DataTableException("Expected an iterable object, got %s" %
                               type(table_description))
    if not isinstance(table_description, dict):
      # We expects a non-dictionary iterable item.
      columns = []
      for desc in table_description:
        parsed_col = DataTable.ColumnTypeParser(desc)
        parsed_col["depth"] = depth
        parsed_col["container"] = "iter"
        columns.append(parsed_col)
      if not columns:
        raise DataTableException("Description iterable objects should not"
                                 " be empty.")
      return columns
    # The other case is a dictionary
    if not table_description:
      raise DataTableException("Empty dictionaries are not allowed inside"
                               " description")

    # To differentiate between the two cases of more levels below or this is
    # the most inner dictionary, we consider the number of keys (more then one
    # key is indication for most inner dictionary) and the type of the key and
    # value in case of only 1 key (if the type of key is string and the type of
    # the value is a tuple of 0-3 items, we assume this is the most inner
    # dictionary).
    # NOTE: this way of differentiating might create ambiguity. See docs.
    if (len(table_description) != 1 or
        (isinstance(next(six.iterkeys(table_description)), six.string_types) and
         isinstance(next(six.itervalues(table_description)), tuple) and
         len(next(six.itervalues(table_description))) < 4)):
      # This is the most inner dictionary. Parsing types.
      columns = []
      # We sort the items, equivalent to sort the keys since they are unique
      for key, value in sorted(table_description.items()):
        # We parse the column type as (key, type) or (key, type, label) using
        # ColumnTypeParser.
        if isinstance(value, tuple):
          parsed_col = DataTable.ColumnTypeParser((key,) + value)
        else:
          parsed_col = DataTable.ColumnTypeParser((key, value))
        parsed_col["depth"] = depth
        parsed_col["container"] = "dict"
        columns.append(parsed_col)
      return columns
    # This is an outer dictionary, must have at most one key.
    parsed_col = DataTable.ColumnTypeParser(sorted(table_description.keys())[0])
    parsed_col["depth"] = depth
    parsed_col["container"] = "dict"
    return ([parsed_col] + DataTable.TableDescriptionParser(
        sorted(table_description.values())[0], depth=depth + 1))

  @property
  def columns(self):
    """Returns the parsed table description."""
    return self.__columns

  def NumberOfRows(self):
    """Returns the number of rows in the current data stored in the table."""
    return len(self.__data)

  def SetRowsCustomProperties(self, rows, custom_properties):
    """Sets the custom properties for given row(s).

    Can accept a single row or an iterable of rows.
    Sets the given custom properties for all specified rows.

    Args:
      rows: The row, or rows, to set the custom properties for.
      custom_properties: A string to string dictionary of custom properties to
      set for all rows.
    """
    if not hasattr(rows, "__iter__"):
      rows = [rows]
    for row in rows:
      self.__data[row] = (self.__data[row][0], custom_properties)

  def LoadData(self, data, custom_properties=None):
    """Loads new rows to the data table, clearing existing rows.

    May also set the custom_properties for the added rows. The given custom
    properties dictionary specifies the dictionary that will be used for *all*
    given rows.

    Args:
      data: The rows that the table will contain.
      custom_properties: A dictionary of string to string to set as the custom
                         properties for all rows.
    """
    self.__data = []
    self.AppendData(data, custom_properties)

  def AppendData(self, data, custom_properties=None):
    """Appends new data to the table.

    Data is appended in rows. Data must comply with
    the table schema passed in to __init__(). See CoerceValue() for a list
    of acceptable data types. See the class documentation for more information
    and examples of schema and data values.

    Args:
      data: The row to add to the table. The data must conform to the table
            description format.
      custom_properties: A dictionary of string to string, representing the
                         custom properties to add to all the rows.

    Raises:
      DataTableException: The data structure does not match the description.
    """
    # If the maximal depth is 0, we simply iterate over the data table
    # lines and insert them using _InnerAppendData. Otherwise, we simply
    # let the _InnerAppendData handle all the levels.
    if not self.__columns[-1]["depth"]:
      for row in data:
        self._InnerAppendData(({}, custom_properties), row, 0)
    else:
      self._InnerAppendData(({}, custom_properties), data, 0)

  def _InnerAppendData(self, prev_col_values, data, col_index):
    """Inner function to assist LoadData."""
    # We first check that col_index has not exceeded the columns size
    if col_index >= len(self.__columns):
      raise DataTableException("The data does not match description, too deep")

    # Dealing with the scalar case, the data is the last value.
    if self.__columns[col_index]["container"] == "scalar":
      prev_col_values[0][self.__columns[col_index]["id"]] = data
      self.__data.append(prev_col_values)
      return

    if self.__columns[col_index]["container"] == "iter":
      if not hasattr(data, "__iter__") or isinstance(data, dict):
        raise DataTableException("Expected iterable object, got %s" %
                                 type(data))
      # We only need to insert the rest of the columns
      # If there are less items than expected, we only add what there is.
      for value in data:
        if col_index >= len(self.__columns):
          raise DataTableException("Too many elements given in data")
        prev_col_values[0][self.__columns[col_index]["id"]] = value
        col_index += 1
      self.__data.append(prev_col_values)
      return

    # We know the current level is a dictionary, we verify the type.
    if not isinstance(data, dict):
      raise DataTableException("Expected dictionary at current level, got %s" %
                               type(data))
    # We check if this is the last level
    if self.__columns[col_index]["depth"] == self.__columns[-1]["depth"]:
      # We need to add the keys in the dictionary as they are
      for col in self.__columns[col_index:]:
        if col["id"] in data:
          prev_col_values[0][col["id"]] = data[col["id"]]
      self.__data.append(prev_col_values)
      return

    # We have a dictionary in an inner depth level.
    if not data.keys():
      # In case this is an empty dictionary, we add a record with the columns
      # filled only until this point.
      self.__data.append(prev_col_values)
    else:
      for key in sorted(data):
        col_values = dict(prev_col_values[0])
        col_values[self.__columns[col_index]["id"]] = key
        self._InnerAppendData((col_values, prev_col_values[1]),
                              data[key], col_index + 1)

  def _PreparedData(self, order_by=()):
    """Prepares the data for enumeration - sorting it by order_by.

    Args:
      order_by: Optional. Specifies the name of the column(s) to sort by, and
                (optionally) which direction to sort in. Default sort direction
                is asc. Following formats are accepted:
                "string_col_name"  -- For a single key in default (asc) order.
                ("string_col_name", "asc|desc") -- For a single key.
                [("col_1","asc|desc"), ("col_2","asc|desc")] -- For more than
                    one column, an array of tuples of (col_name, "asc|desc").

    Returns:
      The data sorted by the keys given.

    Raises:
      DataTableException: Sort direction not in 'asc' or 'desc'
    """
    if not order_by:
      return self.__data

    sorted_data = self.__data[:]
    if isinstance(order_by, six.string_types) or (
        isinstance(order_by, tuple) and len(order_by) == 2 and
        order_by[1].lower() in ["asc", "desc"]):
      order_by = (order_by,)
    for key in reversed(order_by):
      if isinstance(key, six.string_types):
        sorted_data.sort(key=lambda x: x[0].get(key))
      elif (isinstance(key, (list, tuple)) and len(key) == 2 and
            key[1].lower() in ("asc", "desc")):
        key_func = lambda x: x[0].get(key[0])
        sorted_data.sort(key=key_func, reverse=key[1].lower() != "asc")
      else:
        raise DataTableException("Expected tuple with second value: "
                                 "'asc' or 'desc'")

    return sorted_data

  def ToJSCode(self, name, columns_order=None, order_by=()):
    """Writes the data table as a JS code string.

    This method writes a string of JS code that can be run to
    generate a DataTable with the specified data. Typically used for debugging
    only.

    Args:
      name: The name of the table. The name would be used as the DataTable's
            variable name in the created JS code.
      columns_order: Optional. Specifies the order of columns in the
                     output table. Specify a list of all column IDs in the order
                     in which you want the table created.
                     Note that you must list all column IDs in this parameter,
                     if you use it.
      order_by: Optional. Specifies the name of the column(s) to sort by.
                Passed as is to _PreparedData.

    Returns:
      A string of JS code that, when run, generates a DataTable with the given
      name and the data stored in the DataTable object.
      Example result:
        "var tab1 = new google.visualization.DataTable();
         tab1.addColumn("string", "a", "a");
         tab1.addColumn("number", "b", "b");
         tab1.addColumn("boolean", "c", "c");
         tab1.addRows(10);
         tab1.setCell(0, 0, "a");
         tab1.setCell(0, 1, 1, null, {"foo": "bar"});
         tab1.setCell(0, 2, true);
         ...
         tab1.setCell(9, 0, "c");
         tab1.setCell(9, 1, 3, "3$");
         tab1.setCell(9, 2, false);"

    Raises:
      DataTableException: The data does not match the type.
    """

    encoder = DataTableJSONEncoder()

    if columns_order is None:
      columns_order = [col["id"] for col in self.__columns]
    col_dict = dict([(col["id"], col) for col in self.__columns])

    # We first create the table with the given name
    jscode = "var %s = new google.visualization.DataTable();\n" % name
    if self.custom_properties:
      jscode += "%s.setTableProperties(%s);\n" % (
          name, encoder.encode(self.custom_properties))

    # We add the columns to the table
    for i, col in enumerate(columns_order):
      jscode += "%s.addColumn(%s, %s, %s);\n" % (
          name,
          encoder.encode(col_dict[col]["type"]),
          encoder.encode(col_dict[col]["label"]),
          encoder.encode(col_dict[col]["id"]))
      if col_dict[col]["custom_properties"]:
        jscode += "%s.setColumnProperties(%d, %s);\n" % (
            name, i, encoder.encode(col_dict[col]["custom_properties"]))
    jscode += "%s.addRows(%d);\n" % (name, len(self.__data))

    # We now go over the data and add each row
    for (i, (row, cp)) in enumerate(self._PreparedData(order_by)):
      # We add all the elements of this row by their order
      for (j, col) in enumerate(columns_order):
        if col not in row or row[col] is None:
          continue
        value = self.CoerceValue(row[col], col_dict[col]["type"])
        if isinstance(value, tuple):
          cell_cp = ""
          if len(value) == 3:
            cell_cp = ", %s" % encoder.encode(row[col][2])
          # We have a formatted value or custom property as well
          jscode += ("%s.setCell(%d, %d, %s, %s%s);\n" %
                     (name, i, j,
                      self.EscapeForJSCode(encoder, value[0]),
                      self.EscapeForJSCode(encoder, value[1]), cell_cp))
        else:
          jscode += "%s.setCell(%d, %d, %s);\n" % (
              name, i, j, self.EscapeForJSCode(encoder, value))
      if cp:
        jscode += "%s.setRowProperties(%d, %s);\n" % (
            name, i, encoder.encode(cp))
    return jscode

  def ToHtml(self, columns_order=None, order_by=()):
    """Writes the data table as an HTML table code string.

    Args:
      columns_order: Optional. Specifies the order of columns in the
                     output table. Specify a list of all column IDs in the order
                     in which you want the table created.
                     Note that you must list all column IDs in this parameter,
                     if you use it.
      order_by: Optional. Specifies the name of the column(s) to sort by.
                Passed as is to _PreparedData.

    Returns:
      An HTML table code string.
      Example result (the result is without the newlines):
       <html><body><table border="1">
        <thead><tr><th>a</th><th>b</th><th>c</th></tr></thead>
        <tbody>
         <tr><td>1</td><td>"z"</td><td>2</td></tr>
         <tr><td>"3$"</td><td>"w"</td><td></td></tr>
        </tbody>
       </table></body></html>

    Raises:
      DataTableException: The data does not match the type.
    """
    table_template = "<html><body><table border=\"1\">%s</table></body></html>"
    columns_template = "<thead><tr>%s</tr></thead>"
    rows_template = "<tbody>%s</tbody>"
    row_template = "<tr>%s</tr>"
    header_cell_template = "<th>%s</th>"
    cell_template = "<td>%s</td>"

    if columns_order is None:
      columns_order = [col["id"] for col in self.__columns]
    col_dict = dict([(col["id"], col) for col in self.__columns])

    columns_list = []
    for col in columns_order:
      columns_list.append(header_cell_template %
                          html.escape(col_dict[col]["label"]))
    columns_html = columns_template % "".join(columns_list)

    rows_list = []
    # We now go over the data and add each row
    for row, unused_cp in self._PreparedData(order_by):
      cells_list = []
      # We add all the elements of this row by their order
      for col in columns_order:
        # For empty string we want empty quotes ("").
        value = ""
        if col in row and row[col] is not None:
          value = self.CoerceValue(row[col], col_dict[col]["type"])
        if isinstance(value, tuple):
          # We have a formatted value and we're going to use it
          cells_list.append(cell_template % html.escape(self.ToString(value[1])))
        else:
          cells_list.append(cell_template % html.escape(self.ToString(value)))
      rows_list.append(row_template % "".join(cells_list))
    rows_html = rows_template % "".join(rows_list)

    return table_template % (columns_html + rows_html)

  def ToCsv(self, columns_order=None, order_by=(), separator=","):
    """Writes the data table as a CSV string.

    Output is encoded in UTF-8 because the Python "csv" module can't handle
    Unicode properly according to its documentation.

    Args:
      columns_order: Optional. Specifies the order of columns in the
                     output table. Specify a list of all column IDs in the order
                     in which you want the table created.
                     Note that you must list all column IDs in this parameter,
                     if you use it.
      order_by: Optional. Specifies the name of the column(s) to sort by.
                Passed as is to _PreparedData.
      separator: Optional. The separator to use between the values.

    Returns:
      A CSV string representing the table.
      Example result:
       'a','b','c'
       1,'z',2
       3,'w',''

    Raises:
      DataTableException: The data does not match the type.
    """

    csv_buffer = six.StringIO()
    writer = csv.writer(csv_buffer, delimiter=separator)

    if columns_order is None:
      columns_order = [col["id"] for col in self.__columns]
    col_dict = dict([(col["id"], col) for col in self.__columns])

    def ensure_str(s):
      "Compatibility function. Ensures using of str rather than unicode."
      if isinstance(s, str):
        return s
      return s.encode("utf-8")

    writer.writerow([ensure_str(col_dict[col]["label"])
                     for col in columns_order])

    # We now go over the data and add each row
    for row, unused_cp in self._PreparedData(order_by):
      cells_list = []
      # We add all the elements of this row by their order
      for col in columns_order:
        value = ""
        if col in row and row[col] is not None:
          value = self.CoerceValue(row[col], col_dict[col]["type"])
        if isinstance(value, tuple):
          # We have a formatted value. Using it only for date/time types.
          if col_dict[col]["type"] in ["date", "datetime", "timeofday"]:
            cells_list.append(ensure_str(self.ToString(value[1])))
          else:
            cells_list.append(ensure_str(self.ToString(value[0])))
        else:
          cells_list.append(ensure_str(self.ToString(value)))
      writer.writerow(cells_list)
    return csv_buffer.getvalue()

  def ToTsvExcel(self, columns_order=None, order_by=()):
    """Returns a file in tab-separated-format readable by MS Excel.

    Returns a file in UTF-16 little endian encoding, with tabs separating the
    values.

    Args:
      columns_order: Delegated to ToCsv.
      order_by: Delegated to ToCsv.

    Returns:
      A tab-separated little endian UTF16 file representing the table.
    """
    csv_result = self.ToCsv(columns_order, order_by, separator="\t")
    if not isinstance(csv_result, six.text_type):
      csv_result = csv_result.decode("utf-8")
    return csv_result.encode("UTF-16LE")

  def _ToJSonObj(self, columns_order=None, order_by=()):
    """Returns an object suitable to be converted to JSON.

    Args:
      columns_order: Optional. A list of all column IDs in the order in which
                     you want them created in the output table. If specified,
                     all column IDs must be present.
      order_by: Optional. Specifies the name of the column(s) to sort by.
                Passed as is to _PreparedData().

    Returns:
      A dictionary object for use by ToJSon or ToJSonResponse.
    """
    if columns_order is None:
      columns_order = [col["id"] for col in self.__columns]
    col_dict = dict([(col["id"], col) for col in self.__columns])

    # Creating the column JSON objects
    col_objs = []
    for col_id in columns_order:
      col_obj = {"id": col_dict[col_id]["id"],
                 "label": col_dict[col_id]["label"],
                 "type": col_dict[col_id]["type"]}
      if col_dict[col_id]["custom_properties"]:
        col_obj["p"] = col_dict[col_id]["custom_properties"]
      col_objs.append(col_obj)

    # Creating the rows jsons
    row_objs = []
    for row, cp in self._PreparedData(order_by):
      cell_objs = []
      for col in columns_order:
        value = self.CoerceValue(row.get(col, None), col_dict[col]["type"])
        if value is None:
          cell_obj = None
        elif isinstance(value, tuple):
          cell_obj = {"v": value[0]}
          if len(value) > 1 and value[1] is not None:
            cell_obj["f"] = value[1]
          if len(value) == 3:
            cell_obj["p"] = value[2]
        else:
          cell_obj = {"v": value}
        cell_objs.append(cell_obj)
      row_obj = {"c": cell_objs}
      if cp:
        row_obj["p"] = cp
      row_objs.append(row_obj)

    json_obj = {"cols": col_objs, "rows": row_objs}
    if self.custom_properties:
      json_obj["p"] = self.custom_properties

    return json_obj

  def ToJSon(self, columns_order=None, order_by=()):
    """Returns a string that can be used in a JS DataTable constructor.

    This method writes a JSON string that can be passed directly into a Google
    Visualization API DataTable constructor. Use this output if you are
    hosting the visualization HTML on your site, and want to code the data
    table in Python. Pass this string into the
    google.visualization.DataTable constructor, e.g,:
      ... on my page that hosts my visualization ...
      google.setOnLoadCallback(drawTable);
      function drawTable() {
        var data = new google.visualization.DataTable(_my_JSon_string, 0.6);
        myTable.draw(data);
      }

    Args:
      columns_order: Optional. Specifies the order of columns in the
                     output table. Specify a list of all column IDs in the order
                     in which you want the table created.
                     Note that you must list all column IDs in this parameter,
                     if you use it.
      order_by: Optional. Specifies the name of the column(s) to sort by.
                Passed as is to _PreparedData().

    Returns:
      A JSon constructor string to generate a JS DataTable with the data
      stored in the DataTable object.
      Example result (the result is without the newlines):
       {cols: [{id:"a",label:"a",type:"number"},
               {id:"b",label:"b",type:"string"},
              {id:"c",label:"c",type:"number"}],
        rows: [{c:[{v:1},{v:"z"},{v:2}]}, c:{[{v:3,f:"3$"},{v:"w"},null]}],
        p:    {'foo': 'bar'}}

    Raises:
      DataTableException: The data does not match the type.
    """

    encoded_response_str = DataTableJSONEncoder().encode(self._ToJSonObj(columns_order, order_by))
    if not isinstance(encoded_response_str, str):
      return encoded_response_str.encode("utf-8")
    return encoded_response_str

  def ToJSonResponse(self, columns_order=None, order_by=(), req_id=0,
                     response_handler="google.visualization.Query.setResponse"):
    """Writes a table as a JSON response that can be returned as-is to a client.

    This method writes a JSON response to return to a client in response to a
    Google Visualization API query. This string can be processed by the calling
    page, and is used to deliver a data table to a visualization hosted on
    a different page.

    Args:
      columns_order: Optional. Passed straight to self.ToJSon().
      order_by: Optional. Passed straight to self.ToJSon().
      req_id: Optional. The response id, as retrieved by the request.
      response_handler: Optional. The response handler, as retrieved by the
          request.

    Returns:
      A JSON response string to be received by JS the visualization Query
      object. This response would be translated into a DataTable on the
      client side.
      Example result (newlines added for readability):
       google.visualization.Query.setResponse({
          'version':'0.6', 'reqId':'0', 'status':'OK',
          'table': {cols: [...], rows: [...]}});

    Note: The URL returning this string can be used as a data source by Google
          Visualization Gadgets or from JS code.
    """

    response_obj = {
        "version": "0.6",
        "reqId": str(req_id),
        "table": self._ToJSonObj(columns_order, order_by),
        "status": "ok"
    }
    encoded_response_str = DataTableJSONEncoder().encode(response_obj)
    if not isinstance(encoded_response_str, str):
      encoded_response_str = encoded_response_str.encode("utf-8")
    return "%s(%s);" % (response_handler, encoded_response_str)

  def ToResponse(self, columns_order=None, order_by=(), tqx=""):
    """Writes the right response according to the request string passed in tqx.

    This method parses the tqx request string (format of which is defined in
    the documentation for implementing a data source of Google Visualization),
    and returns the right response according to the request.
    It parses out the "out" parameter of tqx, calls the relevant response
    (ToJSonResponse() for "json", ToCsv() for "csv", ToHtml() for "html",
    ToTsvExcel() for "tsv-excel") and passes the response function the rest of
    the relevant request keys.

    Args:
      columns_order: Optional. Passed as is to the relevant response function.
      order_by: Optional. Passed as is to the relevant response function.
      tqx: Optional. The request string as received by HTTP GET. Should be in
           the format "key1:value1;key2:value2...". All keys have a default
           value, so an empty string will just do the default (which is calling
           ToJSonResponse() with no extra parameters).

    Returns:
      A response string, as returned by the relevant response function.

    Raises:
      DataTableException: One of the parameters passed in tqx is not supported.
    """
    tqx_dict = {}
    if tqx:
      tqx_dict = dict(opt.split(":") for opt in tqx.split(";"))
    if tqx_dict.get("version", "0.6") != "0.6":
      raise DataTableException(
          "Version (%s) passed by request is not supported."
          % tqx_dict["version"])

    if tqx_dict.get("out", "json") == "json":
      response_handler = tqx_dict.get("responseHandler",
                                      "google.visualization.Query.setResponse")
      return self.ToJSonResponse(columns_order, order_by,
                                 req_id=tqx_dict.get("reqId", 0),
                                 response_handler=response_handler)
    elif tqx_dict["out"] == "html":
      return self.ToHtml(columns_order, order_by)
    elif tqx_dict["out"] == "csv":
      return self.ToCsv(columns_order, order_by)
    elif tqx_dict["out"] == "tsv-excel":
      return self.ToTsvExcel(columns_order, order_by)
    else:
      raise DataTableException(
          "'out' parameter: '%s' is not supported" % tqx_dict["out"])
