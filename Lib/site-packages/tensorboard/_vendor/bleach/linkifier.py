from __future__ import unicode_literals
import re

from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer

from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes


#: List of default callbacks
DEFAULT_CALLBACKS = [linkify_callbacks.nofollow]


TLDS = """ac ad ae aero af ag ai al am an ao aq ar arpa as asia at au aw ax az
       ba bb bd be bf bg bh bi biz bj bm bn bo br bs bt bv bw by bz ca cat
       cc cd cf cg ch ci ck cl cm cn co com coop cr cu cv cx cy cz de dj dk
       dm do dz ec edu ee eg er es et eu fi fj fk fm fo fr ga gb gd ge gf gg
       gh gi gl gm gn gov gp gq gr gs gt gu gw gy hk hm hn hr ht hu id ie il
       im in info int io iq ir is it je jm jo jobs jp ke kg kh ki km kn kp
       kr kw ky kz la lb lc li lk lr ls lt lu lv ly ma mc md me mg mh mil mk
       ml mm mn mo mobi mp mq mr ms mt mu museum mv mw mx my mz na name nc ne
       net nf ng ni nl no np nr nu nz om org pa pe pf pg ph pk pl pm pn post
       pr pro ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sj sk sl
       sm sn so sr ss st su sv sx sy sz tc td tel tf tg th tj tk tl tm tn to
       tp tr travel tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws
       xn xxx ye yt yu za zm zw""".split()

# Make sure that .com doesn't get matched by .co first
TLDS.reverse()


def build_url_re(tlds=TLDS, protocols=allowed_protocols):
    """Builds the url regex used by linkifier

   If you want a different set of tlds or allowed protocols, pass those in
   and stomp on the existing ``url_re``::

       from bleach import linkifier

       my_url_re = linkifier.build_url_re(my_tlds_list, my_protocols)

       linker = LinkifyFilter(url_re=my_url_re)

    """
    return re.compile(
        r"""\(*  # Match any opening parentheses.
        \b(?<![@.])(?:(?:{0}):/{{0,3}}(?:(?:\w+:)?\w+@)?)?  # http://
        ([\w-]+\.)+(?:{1})(?:\:[0-9]+)?(?!\.\w)\b   # xx.yy.tld(:##)?
        (?:[/?][^\s\{{\}}\|\\\^\[\]`<>"]*)?
            # /path/zz (excluding "unsafe" chars from RFC 1738,
            # except for # and ~, which happen in practice)
        """.format('|'.join(protocols), '|'.join(tlds)),
        re.IGNORECASE | re.VERBOSE | re.UNICODE)


URL_RE = build_url_re()


PROTO_RE = re.compile(r'^[\w-]+:/{0,3}', re.IGNORECASE)


EMAIL_RE = re.compile(
    r"""(?<!//)
    (([-!#$%&'*+/=?^_`{}|~0-9A-Z]+
        (\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*  # dot-atom
    |^"([\001-\010\013\014\016-\037!#-\[\]-\177]
        |\\[\001-\011\013\014\016-\177])*"  # quoted-string
    )@(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6})  # domain
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE)


class Linker(object):
    """Convert URL-like strings in an HTML fragment to links

    This function converts strings that look like URLs, domain names and email
    addresses in text that may be an HTML fragment to links, while preserving:

    1. links already in the string
    2. urls found in attributes
    3. email addresses

    linkify does a best-effort approach and tries to recover from bad
    situations due to crazy text.

    """
    def __init__(self, callbacks=DEFAULT_CALLBACKS, skip_tags=None, parse_email=False,
                 url_re=URL_RE, email_re=EMAIL_RE):
        """Creates a Linker instance

        :arg list callbacks: list of callbacks to run when adjusting tag attributes;
            defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

        :arg list skip_tags: list of tags that you don't want to linkify the
            contents of; for example, you could set this to ``['pre']`` to skip
            linkifying contents of ``pre`` tags

        :arg bool parse_email: whether or not to linkify email addresses

        :arg re url_re: url matching regex

        :arg re email_re: email matching regex

        :returns: linkified text as unicode

        """
        self.callbacks = callbacks
        self.skip_tags = skip_tags
        self.parse_email = parse_email
        self.url_re = url_re
        self.email_re = email_re

        self.parser = html5lib.HTMLParser(namespaceHTMLElements=False)
        self.walker = html5lib.getTreeWalker('etree')
        self.serializer = HTMLSerializer(
            quote_attr_values='always',
            omit_optional_tags=False,

            # linkify does not sanitize
            sanitize=False,

            # linkify alphabetizes
            alphabetical_attributes=False,
        )

    def linkify(self, text):
        """Linkify specified text

        :arg str text: the text to add links to

        :returns: linkified text as unicode

        """
        text = force_unicode(text)

        if not text:
            return u''

        dom = self.parser.parseFragment(text)
        filtered = LinkifyFilter(
            source=self.walker(dom),
            callbacks=self.callbacks,
            skip_tags=self.skip_tags,
            parse_email=self.parse_email,
            url_re=self.url_re,
            email_re=self.email_re,
        )
        return self.serializer.render(filtered)


class LinkifyFilter(Filter):
    """html5lib filter that linkifies text

    This will do the following:

    * convert email addresses into links
    * convert urls into links
    * edit existing links by running them through callbacks--the default is to
      add a ``rel="nofollow"``

    This filter can be used anywhere html5lib filters can be used.

    """
    def __init__(self, source, callbacks=None, skip_tags=None, parse_email=False,
                 url_re=URL_RE, email_re=EMAIL_RE):
        """Creates a LinkifyFilter instance

        :arg TreeWalker source: stream

        :arg list callbacks: list of callbacks to run when adjusting tag attributes;
            defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

        :arg list skip_tags: list of tags that you don't want to linkify the
            contents of; for example, you could set this to ``['pre']`` to skip
            linkifying contents of ``pre`` tags

        :arg bool parse_email: whether or not to linkify email addresses

        :arg re url_re: url matching regex

        :arg re email_re: email matching regex

        """
        super(LinkifyFilter, self).__init__(source)

        self.callbacks = callbacks or []
        self.skip_tags = skip_tags or []
        self.parse_email = parse_email

        self.url_re = url_re
        self.email_re = email_re

    def apply_callbacks(self, attrs, is_new):
        """Given an attrs dict and an is_new bool, runs through callbacks

        Callbacks can return an adjusted attrs dict or ``None``. In the case of
        ``None``, we stop going through callbacks and return that and the link
        gets dropped.

        :arg dict attrs: map of ``(namespace, name)`` -> ``value``

        :arg bool is_new: whether or not this link was added by linkify

        :returns: adjusted attrs dict or ``None``

        """
        for cb in self.callbacks:
            attrs = cb(attrs, is_new)
            if attrs is None:
                return None
        return attrs

    def extract_character_data(self, token_list):
        """Extracts and squashes character sequences in a token stream"""
        # FIXME(willkg): This is a terrible idea. What it does is drop all the
        # tags from the token list and merge the Characters and SpaceCharacters
        # tokens into a single text.
        #
        # So something like this::
        #
        #     "<span>" "<b>" "some text" "</b>" "</span>"
        #
        # gets converted to "some text".
        #
        # This gets used to figure out the ``_text`` fauxttribute value for
        # linkify callables.
        #
        # I'm not really sure how else to support that ``_text`` fauxttribute and
        # maintain some modicum of backwards compatability with previous versions
        # of Bleach.

        out = []
        for token in token_list:
            token_type = token['type']
            if token_type in ['Characters', 'SpaceCharacters']:
                out.append(token['data'])

        return u''.join(out)

    def handle_email_addresses(self, src_iter):
        """Handle email addresses in character tokens"""
        for token in src_iter:
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0

                # For each email address we find in the text
                for match in self.email_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append(
                            {u'type': u'Characters', u'data': text[end:match.start()]}
                        )

                    # Run attributes through the callbacks to see what we
                    # should do with this match
                    attrs = {
                        (None, u'href'): u'mailto:%s' % match.group(0),
                        u'_text': match.group(0)
                    }
                    attrs = self.apply_callbacks(attrs, True)

                    if attrs is None:
                        # Just add the text--but not as a link
                        new_tokens.append(
                            {u'type': u'Characters', u'data': match.group(0)}
                        )

                    else:
                        # Add an "a" tag for the new link
                        _text = attrs.pop(u'_text', '')
                        attrs = alphabetize_attributes(attrs)
                        new_tokens.extend([
                            {u'type': u'StartTag', u'name': u'a', u'data': attrs},
                            {u'type': u'Characters', u'data': force_unicode(_text)},
                            {u'type': u'EndTag', u'name': 'a'}
                        ])
                    end = match.end()

                if new_tokens:
                    # Yield the adjusted set of tokens and then continue
                    # through the loop
                    if end < len(text):
                        new_tokens.append({u'type': u'Characters', u'data': text[end:]})

                    for new_token in new_tokens:
                        yield new_token

                    continue

            yield token

    def strip_non_url_bits(self, fragment):
        """Strips non-url bits from the url

        This accounts for over-eager matching by the regex.

        """
        prefix = suffix = ''

        while fragment:
            # Try removing ( from the beginning and, if it's balanced, from the
            # end, too
            if fragment.startswith(u'('):
                prefix = prefix + u'('
                fragment = fragment[1:]

                if fragment.endswith(u')'):
                    suffix = u')' + suffix
                    fragment = fragment[:-1]
                continue

            # Now try extraneous things from the end. For example, sometimes we
            # pick up ) at the end of a url, but the url is in a parenthesized
            # phrase like:
            #
            #     "i looked at the site (at http://example.com)"

            if fragment.endswith(u')') and u'(' not in fragment:
                fragment = fragment[:-1]
                suffix = u')' + suffix
                continue

            # Handle commas
            if fragment.endswith(u','):
                fragment = fragment[:-1]
                suffix = u',' + suffix
                continue

            # Handle periods
            if fragment.endswith(u'.'):
                fragment = fragment[:-1]
                suffix = u'.' + suffix
                continue

            # Nothing matched, so we're done
            break

        return fragment, prefix, suffix

    def handle_links(self, src_iter):
        """Handle links in character tokens"""
        for token in src_iter:
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0

                for match in self.url_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append(
                            {u'type': u'Characters', u'data': text[end:match.start()]}
                        )

                    url = match.group(0)
                    prefix = suffix = ''

                    # Sometimes we pick up too much in the url match, so look for
                    # bits we should drop and remove them from the match
                    url, prefix, suffix = self.strip_non_url_bits(url)

                    # If there's no protocol, add one
                    if PROTO_RE.search(url):
                        href = url
                    else:
                        href = u'http://%s' % url

                    attrs = {
                        (None, u'href'): href,
                        u'_text': url
                    }
                    attrs = self.apply_callbacks(attrs, True)

                    if attrs is None:
                        # Just add the text
                        new_tokens.append(
                            {u'type': u'Characters', u'data': prefix + url + suffix}
                        )

                    else:
                        # Add the "a" tag!
                        if prefix:
                            new_tokens.append(
                                {u'type': u'Characters', u'data': prefix}
                            )

                        _text = attrs.pop(u'_text', '')
                        attrs = alphabetize_attributes(attrs)

                        new_tokens.extend([
                            {u'type': u'StartTag', u'name': u'a', u'data': attrs},
                            {u'type': u'Characters', u'data': force_unicode(_text)},
                            {u'type': u'EndTag', u'name': 'a'},
                        ])

                        if suffix:
                            new_tokens.append(
                                {u'type': u'Characters', u'data': suffix}
                            )

                    end = match.end()

                if new_tokens:
                    # Yield the adjusted set of tokens and then continue
                    # through the loop
                    if end < len(text):
                        new_tokens.append({u'type': u'Characters', u'data': text[end:]})

                    for new_token in new_tokens:
                        yield new_token

                    continue

            yield token

    def handle_a_tag(self, token_buffer):
        """Handle the "a" tag

        This could adjust the link or drop it altogether depending on what the
        callbacks return.

        This yields the new set of tokens.

        """
        a_token = token_buffer[0]
        if a_token['data']:
            attrs = a_token['data']
        else:
            attrs = {}
        text = self.extract_character_data(token_buffer)
        attrs['_text'] = text

        attrs = self.apply_callbacks(attrs, False)

        if attrs is None:
            # We're dropping the "a" tag and everything else and replacing
            # it with character data. So emit that token.
            yield {'type': 'Characters', 'data': text}

        else:
            new_text = attrs.pop('_text', '')
            a_token['data'] = alphabetize_attributes(attrs)

            if text == new_text:
                # The callbacks didn't change the text, so we yield the new "a"
                # token, then whatever else was there, then the end "a" token
                yield a_token
                for mem in token_buffer[1:]:
                    yield mem

            else:
                # If the callbacks changed the text, then we're going to drop
                # all the tokens between the start and end "a" tags and replace
                # it with the new text
                yield a_token
                yield {'type': 'Characters', 'data': force_unicode(new_text)}
                yield token_buffer[-1]

    def __iter__(self):
        in_a = False
        in_skip_tag = None

        token_buffer = []

        for token in super(LinkifyFilter, self).__iter__():
            if in_a:
                # Handle the case where we're in an "a" tag--we want to buffer tokens
                # until we hit an end "a" tag.
                if token['type'] == 'EndTag' and token['name'] == 'a':
                    # Add the end tag to the token buffer and then handle them
                    # and yield anything returned
                    token_buffer.append(token)
                    for new_token in self.handle_a_tag(token_buffer):
                        yield new_token

                    # Clear "a" related state and continue since we've yielded all
                    # the tokens we're going to yield
                    in_a = False
                    token_buffer = []
                    continue

                else:
                    token_buffer.append(token)
                    continue

            elif token['type'] in ['StartTag', 'EmptyTag']:
                if token['name'] in self.skip_tags:
                    # Skip tags start a "special mode" where we don't linkify
                    # anything until the end tag.
                    in_skip_tag = token['name']

                elif token['name'] == 'a':
                    # The "a" tag is special--we switch to a slurp mode and
                    # slurp all the tokens until the end "a" tag and then
                    # figure out what to do with them there.
                    in_a = True
                    token_buffer.append(token)

                    # We buffer the start tag, so we don't want to yield it,
                    # yet
                    continue

            elif in_skip_tag and self.skip_tags:
                # NOTE(willkg): We put this clause here since in_a and
                # switching in and out of in_a takes precedence.
                if token['type'] == 'EndTag' and token['name'] == in_skip_tag:
                    in_skip_tag = None

            elif not in_a and not in_skip_tag and token['type'] == 'Characters':
                new_stream = iter([token])
                if self.parse_email:
                    new_stream = self.handle_email_addresses(new_stream)

                new_stream = self.handle_links(new_stream)

                for token in new_stream:
                    yield token

                # We've already yielded this token, so continue
                continue

            yield token
