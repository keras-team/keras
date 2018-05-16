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


    # Returns
    A statement related to what the function returns
    This should not be converted: into a list
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

    'arguments': '''- __arg_a__: something related to a
and a new line
- __arg_b__: something related to b
and a sublist:
    - Option one
        - Option one.two
    - Option two
      on a new line
- __arg_c__: nothing to add''',

    'notes': '''Something to note:
- __This_should_be_converted__:
    - a
    - b''',

    'raises': '''- __Exception_a__: if a happens
- __Exception_b__: if b happens''',

    'returns': '''A statement related to what the function returns
This should not be converted: into a list
Neither this.
- __This_should_be_converted__: into a list
    - And this into a sublist
- This_should_not_be_converted: no additional -
- __This_should_be_converted__: with an additional -'''
}


def test_doc_lists():
    doc, block = autogen.process_list_block(test_doc1['doc'], "# Arguments", "$ARGUMENTS$")
    assert test_doc1['arguments'] == block
    doc, block = autogen.process_list_block(test_doc1['doc'], "# Returns", "$RETURNS$")
    assert test_doc1['returns'] == block
    oc, block = autogen.process_list_block(test_doc1['doc'], "# Raises", "$RAISES$")
    assert test_doc1['raises'] == block
    oc, block = autogen.process_list_block(test_doc1['doc'], "# Notes", "$NOTES$")
    assert test_doc1['notes'] == block


if __name__ == '__main__':
    pytest.main([__file__])
