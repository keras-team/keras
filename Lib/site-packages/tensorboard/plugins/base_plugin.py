# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorBoard Plugin abstract base class.

Every plugin in TensorBoard must extend and implement the abstract
methods of this base class.
"""


from abc import ABCMeta
from abc import abstractmethod


class TBPlugin(metaclass=ABCMeta):
    """TensorBoard plugin interface.

    Every plugin must extend from this class.

    Subclasses should have a trivial constructor that takes a TBContext
    argument. Any operation that might throw an exception should either be
    done lazily or made safe with a TBLoader subclass, so the plugin won't
    negatively impact the rest of TensorBoard.

    Fields:
      plugin_name: The plugin_name will also be a prefix in the http
        handlers, e.g. `data/plugins/$PLUGIN_NAME/$HANDLER` The plugin
        name must be unique for each registered plugin, or a ValueError
        will be thrown when the application is constructed. The plugin
        name must only contain characters among [A-Za-z0-9_.-], and must
        be nonempty, or a ValueError will similarly be thrown.
    """

    plugin_name = None

    def __init__(self, context):
        """Initializes this plugin.

        The default implementation does nothing. Subclasses are encouraged
        to override this and save any necessary fields from the `context`.

        Args:
          context: A `base_plugin.TBContext` object.
        """
        pass

    @abstractmethod
    def get_plugin_apps(self):
        """Returns a set of WSGI applications that the plugin implements.

        Each application gets registered with the tensorboard app and is served
        under a prefix path that includes the name of the plugin.

        Returns:
          A dict mapping route paths to WSGI applications. Each route path
          should include a leading slash.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_active(self):
        """Determines whether this plugin is active.

        A plugin may not be active for instance if it lacks relevant data. If a
        plugin is inactive, the frontend may avoid issuing requests to its routes.

        Returns:
          A boolean value. Whether this plugin is active.
        """
        raise NotImplementedError()

    def frontend_metadata(self):
        """Defines how the plugin will be displayed on the frontend.

        The base implementation returns a default value. Subclasses
        should override this and specify either an `es_module_path` or
        (for legacy plugins) an `element_name`, and are encouraged to
        set any other relevant attributes.
        """
        return FrontendMetadata()

    def data_plugin_names(self):
        """Experimental. Lists plugins whose summary data this plugin reads.

        Returns:
          A collection of strings representing plugin names (as read
          from `SummaryMetadata.plugin_data.plugin_name`) from which
          this plugin may read data. Defaults to `(self.plugin_name,)`.
        """
        return (self.plugin_name,)


class FrontendMetadata:
    """Metadata required to render a plugin on the frontend.

    Each argument to the constructor is publicly accessible under a
    field of the same name. See constructor docs for further details.
    """

    def __init__(
        self,
        *,
        disable_reload=None,
        element_name=None,
        es_module_path=None,
        remove_dom=None,
        tab_name=None,
        is_ng_component=None,
    ):
        """Creates a `FrontendMetadata` value.

        The argument list is sorted and may be extended in the future;
        therefore, callers must pass only named arguments to this
        constructor.

        Args:
          disable_reload: Whether to disable the reload button and
              auto-reload timer. A `bool`; defaults to `False`.
          element_name: For legacy plugins, name of the custom element
              defining the plugin frontend: e.g., `"tf-scalar-dashboard"`.
              A `str` or `None` (for iframed plugins). Mutually exclusive
              with `es_module_path`.
          es_module_path: ES module to use as an entry point to this plugin.
              A `str` that is a key in the result of `get_plugin_apps()`, or
              `None` for legacy plugins bundled with TensorBoard as part of
              `webfiles.zip`. Mutually exclusive with legacy `element_name`
          remove_dom: Whether to remove the plugin DOM when switching to a
              different plugin, to trigger the Polymer 'detached' event.
              A `bool`; defaults to `False`.
          tab_name: Name to show in the menu item for this dashboard within
              the navigation bar. May differ from the plugin name: for
              instance, the tab name should not use underscores to separate
              words. Should be a `str` or `None` (the default; indicates to
              use the plugin name as the tab name).
          is_ng_component: Set to `True` only for built-in Angular plugins.
              In this case, the `plugin_name` property of the Plugin, which is
              mapped to the `id` property in JavaScript's `UiPluginMetadata` type,
              is used to select the Angular component. A `True` value is mutually
              exclusive with `element_name` and `es_module_path`.
        """
        self._disable_reload = (
            False if disable_reload is None else disable_reload
        )
        self._element_name = element_name
        self._es_module_path = es_module_path
        self._remove_dom = False if remove_dom is None else remove_dom
        self._tab_name = tab_name
        self._is_ng_component = (
            False if is_ng_component is None else is_ng_component
        )

    @property
    def disable_reload(self):
        return self._disable_reload

    @property
    def element_name(self):
        return self._element_name

    @property
    def is_ng_component(self):
        return self._is_ng_component

    @property
    def es_module_path(self):
        return self._es_module_path

    @property
    def remove_dom(self):
        return self._remove_dom

    @property
    def tab_name(self):
        return self._tab_name

    def __eq__(self, other):
        if not isinstance(other, FrontendMetadata):
            return False
        if self._disable_reload != other._disable_reload:
            return False
        if self._disable_reload != other._disable_reload:
            return False
        if self._element_name != other._element_name:
            return False
        if self._es_module_path != other._es_module_path:
            return False
        if self._remove_dom != other._remove_dom:
            return False
        if self._tab_name != other._tab_name:
            return False
        return True

    def __hash__(self):
        return hash(
            (
                self._disable_reload,
                self._element_name,
                self._es_module_path,
                self._remove_dom,
                self._tab_name,
                self._is_ng_component,
            )
        )

    def __repr__(self):
        return "FrontendMetadata(%s)" % ", ".join(
            (
                "disable_reload=%r" % self._disable_reload,
                "element_name=%r" % self._element_name,
                "es_module_path=%r" % self._es_module_path,
                "remove_dom=%r" % self._remove_dom,
                "tab_name=%r" % self._tab_name,
                "is_ng_component=%r" % self._is_ng_component,
            )
        )


class TBContext:
    """Magic container of information passed from TensorBoard core to plugins.

    A TBContext instance is passed to the constructor of a TBPlugin class. Plugins
    are strongly encouraged to assume that any of these fields can be None. In
    cases when a field is considered mandatory by a plugin, it can either crash
    with ValueError, or silently choose to disable itself by returning False from
    its is_active method.

    All fields in this object are thread safe.
    """

    def __init__(
        self,
        *,
        assets_zip_provider=None,
        data_provider=None,
        flags=None,
        logdir=None,
        multiplexer=None,
        plugin_name_to_instance=None,
        sampling_hints=None,
        window_title=None,
    ):
        """Instantiates magic container.

        The argument list is sorted and may be extended in the future; therefore,
        callers must pass only named arguments to this constructor.

        Args:
          assets_zip_provider: A function that returns a newly opened file handle
              for a zip file containing all static assets. The file names inside the
              zip file are considered absolute paths on the web server. The file
              handle this function returns must be closed. It is assumed that you
              will pass this file handle to zipfile.ZipFile. This zip file should
              also have been created by the tensorboard_zip_file build rule.
          data_provider: Instance of `tensorboard.data.provider.DataProvider`. May
            be `None` if `flags.generic_data` is set to `"false"`.
          flags: An object of the runtime flags provided to TensorBoard to their
              values.
          logdir: The string logging directory TensorBoard was started with.
          multiplexer: An EventMultiplexer with underlying TB data. Plugins should
              copy this data over to the database when the db fields are set.
          plugin_name_to_instance: A mapping between plugin name to instance.
              Plugins may use this property to access other plugins. The context
              object is passed to plugins during their construction, so a given
              plugin may be absent from this mapping until it is registered. Plugin
              logic should handle cases in which a plugin is absent from this
              mapping, lest a KeyError is raised.
          sampling_hints: Map from plugin name to `int` or `NoneType`, where
              the value represents the user-specified downsampling limit as
              given to the `--samples_per_plugin` flag, or `None` if none was
              explicitly given for this plugin.
          window_title: A string specifying the window title.
        """
        self.assets_zip_provider = assets_zip_provider
        self.data_provider = data_provider
        self.flags = flags
        self.logdir = logdir
        self.multiplexer = multiplexer
        self.plugin_name_to_instance = plugin_name_to_instance
        self.sampling_hints = sampling_hints
        self.window_title = window_title


class TBLoader:
    """TBPlugin factory base class.

    Plugins can override this class to customize how a plugin is loaded at
    startup. This might entail adding command-line arguments, checking if
    optional dependencies are installed, and potentially also specializing
    the plugin class at runtime.

    When plugins use optional dependencies, the loader needs to be
    specified in its own module. That way it's guaranteed to be
    importable, even if the `TBPlugin` itself can't be imported.

    Subclasses must have trivial constructors.
    """

    def define_flags(self, parser):
        """Adds plugin-specific CLI flags to parser.

        The default behavior is to do nothing.

        When overriding this method, it's recommended that plugins call the
        `parser.add_argument_group(plugin_name)` method for readability. No
        flags should be specified that would cause `parse_args([])` to fail.

        Args:
          parser: The argument parsing object, which may be mutated.
        """
        pass

    def fix_flags(self, flags):
        """Allows flag values to be corrected or validated after parsing.

        Args:
          flags: The parsed argparse.Namespace object.

        Raises:
          base_plugin.FlagsError: If a flag is invalid or a required
              flag is not passed.
        """
        pass

    def load(self, context):
        """Loads a TBPlugin instance during the setup phase.

        Args:
          context: The TBContext instance.

        Returns:
          A plugin instance or None if it could not be loaded. Loaders that return
          None are skipped.

        :type context: TBContext
        :rtype: TBPlugin | None
        """
        return None


class BasicLoader(TBLoader):
    """Simple TBLoader that's sufficient for most plugins."""

    def __init__(self, plugin_class):
        """Creates simple plugin instance maker.

        :param plugin_class: :class:`TBPlugin`
        """
        self.plugin_class = plugin_class

    def load(self, context):
        return self.plugin_class(context)


class FlagsError(ValueError):
    """Raised when a command line flag is not specified or is invalid."""

    pass
