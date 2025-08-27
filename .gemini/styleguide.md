# Style Guide

## General Principles

APIs added should be:
- Scalable
- Robust
- Backwards compatible
- Future-proof
- Interoperable
- Must work across all backends

---

# Keras API design guidelines

These guidelines are meant to help focus design discussions and help us create d
elightful developer experiences.

These are meant as guidelines, not rules: each decision should be debated in its
 own unique context.

Some text remixed from external references:

- [User experience design for APIs](https://blog.keras.io/user-experience-design
-for-apis.html)
- [Notes to Myself on Software Engineering](https://medium.com/s/story/notes-to-
myself-on-software-engineering-c890f16f4e4d)


---

## Design end-to-end workflows, not individual functions and classes.

When developing APIs, start by designing end-to-end workflows, and only sketch o
ut specific function/class signatures at the end.

- The goal is to arrive to workflows that feel like they are purposefully design
ed and well-optimized, rather than cobbled together to route around the features
 provided by the API. The workflows should come first, before atomic features. *
*Features only exist to support a workflow.** No feature should exist to provide
 a capability “just in case”, “because we can”.
- **Every design review document should prominently feature a code example of on
e or two end-to-end workflows showing the canonical use-case for the new API.**
- Every time we discuss choices surrounding a specific API feature, we should st
art by asking: **in what workflows will this be used?** Then we should make the
choice that makes the most sense with respect to these workflows. We should not
make API design decisions about features in isolation.
- This implies that we will often ask the question: **do users really need to co
nfigure this parameter?**, and in many cases, the answer will be “no”, rather th
an being “yes” by default.


---

## Carefully weigh whether a new feature should be included.


It’s okay to say no: just because someone asks for a feature doesn’t mean we sho
uld do it. Every feature has a cost that goes beyond the initial CL: maintenance
 cost, documentation cost, and cognitive cost for our users (a sprawling API sur
face is a major usability issue).

In particular, in the Keras API, every new feature has to be maintained in perpe
tuity, and has to be replicated in every implementation of the Keras API (which
includes tf.keras, tensorflow.js, and other third-party implementations).

As, such, our criteria for adding a new feature in the API is the following:

- **It should be broadly useful to our users**, rather than a niche feature that
 is only relevant to a specific vertical of researchers. Niche features should b
e maintained independently by those who need them (e.g. by extending the API via
 subclassing), as third-party add-on packages.
- **It should be widely recognized as a machine learning best practice.** We wil
l not add new layers/etc that were recently published to ArXiv.org, even in case
 of claims of increased accuracy/etc. We only add new objects that are already c
ommonly used in the machine learning community. Presumably, a new technique that
 does result in meaningful gains would be broadly adopted after a few months any
way (like ResNet), and that’s when we would be adding it to the core API. SIG-ad
dons maintains a repository of significantly more volatile and independently mai
ntained code to which the barriers to entry are lower.
- **It should have an owner committed to maintaining it in the long term.** In p
articular, the code should be maintainable by multiple people on the team, not j
ust by one technical guru.


In addition, when saying yes to a request for supporting a new use case, remembe
r that **literally adding what the user/team requested is often not the optimal
choice**. Users are focused on their own specific use case, and we must counter
this with a holistic and principled vision of the whole project (see: designing
end-to-end workflows, not atomic functions/classes). Often, the right answer is
to extend an existing feature. **Find the natural place to integrate the new fea
ture in existing APIs.**


### Examples:

- We should not have added the self-normalizing activation function to the API.
It was added before passing the test of time, and that technique has shown later
 not to reach broad adoption. **Note that citation count is not a good metric of
 adoption**; that paper has a high citation count.
- We should not move to core an API that has debuted somewhere on GitHub or TF-A
ddons but has failed to gain more than a few users after a few months.


---

## Seek to minimize cognitive load for our users.

Always seek to minimize the cognitive load imposed on our users in the course of
 using our APIs.

At a high level:

- **Automate everything that can be automated.**
- **Minimize the actions & choices required from the user.** Make sure default v
alues for arguments are sensible and reflect best practices (so that users usual
ly wouldn’t have to manually configure these). Don’t expose options that are not
 important or do not match real use cases, “just in case”.
- **Design simple and consistent workflows that reflect simple and consistent me
ntal models.**

Here are a few practical rules:

- **No API should deal with internal implementation details.** An API is a langu
age for our users to talk about the problem they care about -- and they don’t ca
re about our internal hacks. For instance, an option like `use_locking` in an op
timizer should be avoided. If an argument requires users to understand the imple
mentation (not just what the code is supposed to implement, like SGD in this cas
e), then the argument should not be included in the public API. **An API is all
about the problem it solves, not about how the code works in the background.**
- **Introduce as few new concepts as possible.** It's not just that additional d
ata structures require more effort in order to learn about their methods and pro
perties, it's that they multiply the number of **mental models** that are necess
ary to grok your API. Ideally, you should only need **a single universal mental
model around which everything is organized** (in Keras, that's the `Layer`). Def
initely avoid having more than 2 or 3 mental models underlying the workflows you
 design. Likewise, avoid having concepts that are mostly overlapping but subtly
different, since the difference will be difficult to convey clearly and will con
fuse our users (like, say, `Network` and `Model` -- this is why we don't export
`Network` as a public API).
- **Objects that do interchangeable things should have identical or very close A
PIs.** In particular they should have the same positional arguments. For example
, it should be possible to swap one optimizer for another in user code (when lea
ving all arguments to their default value) without editing the arguments.
- **If you find yourself proposing a signature with more than 6-7 arguments, con
sider whether all of these arguments are useful.** How many people and use cases
 would be affected if you removed one argument? How much would they be affected
-- would they be able to easily extend the API (e.g. via subclassing) to support
 their use case without that built-in argument? Could this API be broken up into
 smaller, modular objects?
- **Best-practices should come baked into your API.** The simplest way to use yo
ur API (leaving all arguments to their default value, using the most obvious too
l for the task, etc) should be as close as possible to the best way of solving t
he problem. In particular, all arguments that can be given a default value shoul
d be given a default value, and that default should match the most common use ca
se.
- **Plain Python types are preferable to custom types.** Use tuples, strings, in
ts... A custom type requires more knowledge and effort on the part of the user (
e.g. `TensorShape`, which is also breaking established conventions of scientific
 Python). **When using enums, make sure that their values are strings**, so as t
o make it possible for users to pass plain strings (example: `data_format="chann
els_last"`, `padding="valid"`).
- **Explicit, single-level configuration arguments are preferable to nested, hid
den configuration arguments.** Avoid something like: `MyLayer(hyperparameter_dic
t)`, instead use `MyLayer(units, activation=None, ...)`.
- **No API should rely on TF Variable names or Op names.** These change all the
time, and should be considered a convenience, not a part of the TensorFlow & Ker
as API.

In particular, naming is important and difficult:

- **The meaning of an argument should be clear from its name and should not requ
ire knowledge that only the implementers have.** In particular, argument names s
hould only involve recognized terms of art (“L1 norm” is a term of art), and sho
uld not involve implementation-related vocabulary (e.g. “fused batchnorm”).
- **Avoid `OverlyLongAndSpecificNamingPatterns`.** If you find yourself with arg
ument names with involve more than 3 subparts (e.g. “squared_operator_norm”), re
consider. Argument names should be intuitive and easy to remember.
- Avoid overly generic names (`x`, `variable`, `parameter`).
- **Make sure you are consistent in your naming choices.** Naming consistency me
ans both **internal naming consistency** (don’t call `dim` what is called `axis`
 in other places, don’t call `ndims` what is called `ndim` elsewhere) and **cons
istency with established conventions for the problem domain (terms of art)**. Be
fore settling on a name, make sure to look up existing names used by domain expe
rts (or other APIs). In our case, argument names should be consistent with the b
roader scientific Python conventions, in particular NumPy.

Note that Keras uses the following naming rules:

- We use the convention `num_*` for counters, though omitting an explicit counte
r is nicer when there is no ambiguity (e.g. `units`, `epochs`, `filters`).
- The rank of a tensor is its `ndim`. A specific dimension index is an `axis`. T
he number of dimensions in a linear projection (or similar) is `units`.
- By convention Keras layers are named with nouns rather than verbs (e.g. `Norma
lization` and not `Normalize`, `Convolution` and not `Convolve`).
- Following Python conventions, classes use capitalized parts (e.g. `ClassName`)
 and functions and methods use snake case (e.g. `function_name`).
- If an argument name has a numerical suffix (e.g. `alpha_1`), we put an undersc
ore before the suffix in snake case. The capitalized equivalent would be e.g. `A
lpha1`.
- We used fully spelled-out names, e.g. `attention_scores` and not `attn_scores`
. There are a couple standardized exceptions to this rule, in particular `dim` f
or "dimension" and `num` for "number". These are sufficiently common that they a
re not ambiguous to a first-time reader.


### Example:

```python
MyConstructor(
   per_variable_sparsity_config=[
      'layer_1/kernel:0.8', 'layer_2/kernel:1.5'])
```

What's wrong with this?

- Overly long argument name
- Too much cognitive load involved in preparing an appropriate argument value
- Preparing an argument value requires internal implementation knowledge
- Reliance on TF variable names (subject to changes at any time, thus breaking t
his code)
- Nested config adding indirection
- Incorrect typing (float values being passing as strings)

Possible alternative:

```
obj = MyConstructor()
obj.configure_sparsity(some_layer.kernel, value=0.8)
obj.configure_sparsity(some_other_layer.kernel, value=1.5)
```

What's nice about this?

- Object-based variable references.
- Modular, simple action, with a clear name.
- Plain Python types.


---

## Balance expressivity vs. user-friendliness.

### Simple use cases should be simple, advanced use cases should be possible:

**Don’t increase the cognitive load of common use cases for the sake of niche us
e cases**, even minimally.
**Make sure that advanced users have a path to support their use case**, even if
 this path requires the users to roll out plugins or other API extensions (in pa
rticular via subclassing). **It is ok for advanced use cases not to be directly
supported in the built-in API options.**


### Keep our APIs modular.

**Complex objects should be achievable by composing simple objects with few argu
ments, that do one thing reliably.** There is a balance to strike between having
 complex signatures on fewer objects, and having more objects with simpler signa
tures. A good API has a reasonable number of objects, with reasonably simple sig
natures (see also: avoiding signatures with more than 6-7 arguments).

**Things that create state or side-effects should be classes. Functions should b
e stateless.**
For instance, layers that create weights should not be cast as functions, since
it makes the weights (and other elements of state) hard to access, impossible to
 update, and forces reliance on a global state capturing the side effects of lay
er-functions.


### APIs should be strictly compartmentalized.

For instance, the optimizer API or the layers API should not contain arguments f
or configuring distributed training. That should go into the distribution API.


---

## Don’t neglect error messages, docstrings, and documentation.

Documentation and error messages are an integral part of the API. Good docs and
helpful error messages are key to a delightful user experience.

- **Catch user errors early and anticipate common mistakes.** Do user input vali
dation as soon as possible. Actively keep track of common mistakes that people m
ake (by screening GitHub and StackOverflow), and either solve them by simplifyin
g our API, adding targeted error messages for these mistakes, or having a "solut
ions to common issues" page in our docs. Consider adding automated fallback beha
viors (e.g. casting a wrongly-typed input) instead of raising errors, when appli
cable. Be nice to our users.
- **Provide detailed feedback messages upon user error.** Error messages should
be contextual, informative, and actionable. Every error message that transparent
ly provides the user with the solution to their problem means one less support t
icket, multiplied by how many times users run into the same issue. A good error
message should answer:
    - What happened, in what context?
    - What did the software expect?
    - How can the user fix it?
- **A docstring should answer the question: what is this about, and why & how sh
ould I use it?** It should assume as little context as possible, and it shouldn
’t mention specialized terms without first introducing them (for example, “num_b
locks: Number of blocks in the kernel” is not a good argument description if thi
s is the first time you mention “blocks” in your docstring).
- **Show, don’t tell: your documentation should not talk about how the software
works, it should show how to use it.** Show code examples for end-to-end workflo
ws; show code examples for each and every common use case and key feature of you
r API. **All docstrings should include code examples.**
- **Deliberately design the user onboarding process for your feature.** How are
complete newcomers going to find out the best way to solve their use case with y
our tool? Have an answer ready. Make sure your onboarding material closely maps
to what your users care about: don't teach newcomers how your framework is imple
mented, teach them how they can use it to solve their own problems. After shippi
ng a CL and writing good docstrings, make sure to create a Colab guide / tutoria
l showcasing the target workflow, and post it on the docs website or the TF blog
.
- The feature is not ready until:
    - 1) Users know about it
    - 2) They know how to use it
    - 3) They're actually using it to solve the corresponding problem.


Note that Keras uses the following rules for writing docstrings:

- For class docstrings, document arguments in a `Arguments:` section in the clas
s docstring, not in `__init__`.
    - When a user creates a class, they are not calling the `MyLayer.__init__()`
 method as if it were a regular method, they are calling `MyLayer`. We don't wan
t to generate documentation for the `__init__()` method as a standalone method t
hat needs to be called directly, that would be confusing. We also don't need `__
init__()` docstrings that always start with "Initializes a MyLayer class.", whic
h is useless information. Leaving `__init__()` without a docstring is the best p
ractice.
    - If constructor arguments are documented in `__init__`, it forces us to pro
grammatically copy the `__init__` docstring when generating docs and concatenate
 it to the class docstring. This means that the Arguments section becomes the la
st thing in the docstring, which is bad.
- The order of information in a class docstring should be:
    - One-line description of the class, that gives initial context to the user.
 e.g. `Applies Dropout to the input.` Make sure the one-line description is usef
ul. No `Intantiates an ObscureName class instance.`
    - Paragraph(s) of more detailed information that tells the user what the obj
ect is for and when they need to use it. e.g. `The Dropout layer randomly sets i
nput units to 0 with a frequency of "rate" at each step during training time, wh
ich helps prevent overfitting. Inputs not set to 0 are scaled up by "1/(1 - rate
)" such that the sum over all inputs is unchanged. [...]`
    - If there is a reference paper, cite it here.
    - `Arguments` section.
    - If it's a layer that has arguments in `call`, the `Call arguments` section
.
    - If it's a `Layer`, `Input shape` and `Output shape` sections.
    - Example(s).
    - Lastly, addendum. Information that isn't very important and that most user
s don't need, but that should be documented somewhere.
        - e.g. the section "About the layer's `dtype` attribute" in the base Lay
er class.
        - e.g. warnings about edge cases or compatibility issues.
        - e.g. pointers to further guides and tutorials.


### Error messages: a case study


The following would be a very poor error message:

```
AssertionError: '1 != 3'
```

In general, to validate user input, always use `ValueError` and avoid `assert`.

Also bad:

```
ValueError: 'Invalid target shape (600, 1).'
```

The following is better, but still not sufficient, because it does not tell the
user what they passed, and does not quite say how to fix it:

```
ValueError: 'categorical_crossentropy requires target.shape[1] == classes'
```

Now, here's a good example, that says **what was passed**, **what was expected**
, and **how to fix the issue**:

```
ValueError: '''You are passing a target array of shape (600, 1) while using as l
oss `categorical_crossentropy`.
`categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of
shape (samples, classes).
If your targets are integer classes, you can convert them to the expected format
 via:

---
from keras.utils import to_categorical
y_binary = to_categorical(y_int)
---

Alternatively, you can use the loss function `sparse_categorical_crossentropy` i
nstead, which does expect integer targets.
```
