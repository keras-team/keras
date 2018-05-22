from docs import autogen
import pytest

test_doc1 = {
    'doc': '''This function does something really interesting.
Like for example foo baz.

    # Arguments
        arg_a: something related to a
        and a new line
        arg_b: something related to b
        and a sublist:
            - Option one
                - Option one.two
            - Option two
              on a new line
        arg_c: nothing to add
        arg_d: with an additional
            list:
            - Option one

    # Returns
    A statement related to what the function returns
    This: should not be converted into a list
        Neither this.
        This_should_be_converted: into a list
            - And this into a sublist
        - This_should_not_be_converted: no additional -
        This_should_be_converted: with an additional -

    # Notes
        Something to note:
        This_should_be_converted:
            - a
            - b

    # Raises
        Exception_a: if a happens
        Exception_b: if b happens''',
    'result': '''This function does something really interesting.
Like for example foo baz.

__Arguments__

- __arg_a__: something related to a
and a new line
- __arg_b__: something related to b
and a sublist:
    - Option one
        - Option one.two
    - Option two
      on a new line
- __arg_c__: nothing to add
- __arg_d__: with an additional
    list:
    - Option one

__Returns__

A statement related to what the function returns
This: should not be converted into a list
Neither this.
- __This_should_be_converted__: into a list
    - And this into a sublist
- This_should_not_be_converted: no additional -
- __This_should_be_converted__: with an additional -

__Notes__

Something to note:
- __This_should_be_converted__:
    - a
    - b

__Raises__

- __Exception_a__: if a happens
- __Exception_b__: if b happens'''}


def test_doc_lists():
    docstring = autogen.process_docstring(test_doc1['doc'])
    assert docstring == test_doc1['result']


if __name__ == '__main__':
    pytest.main([__file__])
