var tbp = (function () {

	var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

	var zone = {};

	var __spreadArray = (commonjsGlobal && commonjsGlobal.__spreadArray) || function (to, from, pack) {
	    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
	        if (ar || !(i in from)) {
	            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
	            ar[i] = from[i];
	        }
	    }
	    return to.concat(ar || Array.prototype.slice.call(from));
	};
	/**
	 * @license Angular v15.1.0-next.0
	 * (c) 2010-2022 Google LLC. https://angular.io/
	 * License: MIT
	 */
	(function (factory) {
	    factory();
	})((function () {
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    ((function (global) {
	        var performance = global['performance'];
	        function mark(name) {
	            performance && performance['mark'] && performance['mark'](name);
	        }
	        function performanceMeasure(name, label) {
	            performance && performance['measure'] && performance['measure'](name, label);
	        }
	        mark('Zone');
	        // Initialize before it's accessed below.
	        // __Zone_symbol_prefix global can be used to override the default zone
	        // symbol prefix with a custom one if needed.
	        var symbolPrefix = global['__Zone_symbol_prefix'] || '__zone_symbol__';
	        function __symbol__(name) {
	            return symbolPrefix + name;
	        }
	        var checkDuplicate = global[__symbol__('forceDuplicateZoneCheck')] === true;
	        if (global['Zone']) {
	            // if global['Zone'] already exists (maybe zone.js was already loaded or
	            // some other lib also registered a global object named Zone), we may need
	            // to throw an error, but sometimes user may not want this error.
	            // For example,
	            // we have two web pages, page1 includes zone.js, page2 doesn't.
	            // and the 1st time user load page1 and page2, everything work fine,
	            // but when user load page2 again, error occurs because global['Zone'] already exists.
	            // so we add a flag to let user choose whether to throw this error or not.
	            // By default, if existing Zone is from zone.js, we will not throw the error.
	            if (checkDuplicate || typeof global['Zone'].__symbol__ !== 'function') {
	                throw new Error('Zone already loaded.');
	            }
	            else {
	                return global['Zone'];
	            }
	        }
	        var Zone = /** @class */ (function () {
	            function Zone(parent, zoneSpec) {
	                this._parent = parent;
	                this._name = zoneSpec ? zoneSpec.name || 'unnamed' : '<root>';
	                this._properties = zoneSpec && zoneSpec.properties || {};
	                this._zoneDelegate =
	                    new _ZoneDelegate(this, this._parent && this._parent._zoneDelegate, zoneSpec);
	            }
	            Zone.assertZonePatched = function () {
	                if (global['Promise'] !== patches['ZoneAwarePromise']) {
	                    throw new Error('Zone.js has detected that ZoneAwarePromise `(window|global).Promise` ' +
	                        'has been overwritten.\n' +
	                        'Most likely cause is that a Promise polyfill has been loaded ' +
	                        'after Zone.js (Polyfilling Promise api is not necessary when zone.js is loaded. ' +
	                        'If you must load one, do so before loading zone.js.)');
	                }
	            };
	            Object.defineProperty(Zone, "root", {
	                get: function () {
	                    var zone = Zone.current;
	                    while (zone.parent) {
	                        zone = zone.parent;
	                    }
	                    return zone;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Object.defineProperty(Zone, "current", {
	                get: function () {
	                    return _currentZoneFrame.zone;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Object.defineProperty(Zone, "currentTask", {
	                get: function () {
	                    return _currentTask;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            // tslint:disable-next-line:require-internal-with-underscore
	            Zone.__load_patch = function (name, fn, ignoreDuplicate) {
	                if (ignoreDuplicate === void 0) { ignoreDuplicate = false; }
	                if (patches.hasOwnProperty(name)) {
	                    // `checkDuplicate` option is defined from global variable
	                    // so it works for all modules.
	                    // `ignoreDuplicate` can work for the specified module
	                    if (!ignoreDuplicate && checkDuplicate) {
	                        throw Error('Already loaded patch: ' + name);
	                    }
	                }
	                else if (!global['__Zone_disable_' + name]) {
	                    var perfName = 'Zone:' + name;
	                    mark(perfName);
	                    patches[name] = fn(global, Zone, _api);
	                    performanceMeasure(perfName, perfName);
	                }
	            };
	            Object.defineProperty(Zone.prototype, "parent", {
	                get: function () {
	                    return this._parent;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Object.defineProperty(Zone.prototype, "name", {
	                get: function () {
	                    return this._name;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Zone.prototype.get = function (key) {
	                var zone = this.getZoneWith(key);
	                if (zone)
	                    return zone._properties[key];
	            };
	            Zone.prototype.getZoneWith = function (key) {
	                var current = this;
	                while (current) {
	                    if (current._properties.hasOwnProperty(key)) {
	                        return current;
	                    }
	                    current = current._parent;
	                }
	                return null;
	            };
	            Zone.prototype.fork = function (zoneSpec) {
	                if (!zoneSpec)
	                    throw new Error('ZoneSpec required!');
	                return this._zoneDelegate.fork(this, zoneSpec);
	            };
	            Zone.prototype.wrap = function (callback, source) {
	                if (typeof callback !== 'function') {
	                    throw new Error('Expecting function got: ' + callback);
	                }
	                var _callback = this._zoneDelegate.intercept(this, callback, source);
	                var zone = this;
	                return function () {
	                    return zone.runGuarded(_callback, this, arguments, source);
	                };
	            };
	            Zone.prototype.run = function (callback, applyThis, applyArgs, source) {
	                _currentZoneFrame = { parent: _currentZoneFrame, zone: this };
	                try {
	                    return this._zoneDelegate.invoke(this, callback, applyThis, applyArgs, source);
	                }
	                finally {
	                    _currentZoneFrame = _currentZoneFrame.parent;
	                }
	            };
	            Zone.prototype.runGuarded = function (callback, applyThis, applyArgs, source) {
	                if (applyThis === void 0) { applyThis = null; }
	                _currentZoneFrame = { parent: _currentZoneFrame, zone: this };
	                try {
	                    try {
	                        return this._zoneDelegate.invoke(this, callback, applyThis, applyArgs, source);
	                    }
	                    catch (error) {
	                        if (this._zoneDelegate.handleError(this, error)) {
	                            throw error;
	                        }
	                    }
	                }
	                finally {
	                    _currentZoneFrame = _currentZoneFrame.parent;
	                }
	            };
	            Zone.prototype.runTask = function (task, applyThis, applyArgs) {
	                if (task.zone != this) {
	                    throw new Error('A task can only be run in the zone of creation! (Creation: ' +
	                        (task.zone || NO_ZONE).name + '; Execution: ' + this.name + ')');
	                }
	                // https://github.com/angular/zone.js/issues/778, sometimes eventTask
	                // will run in notScheduled(canceled) state, we should not try to
	                // run such kind of task but just return
	                if (task.state === notScheduled && (task.type === eventTask || task.type === macroTask)) {
	                    return;
	                }
	                var reEntryGuard = task.state != running;
	                reEntryGuard && task._transitionTo(running, scheduled);
	                task.runCount++;
	                var previousTask = _currentTask;
	                _currentTask = task;
	                _currentZoneFrame = { parent: _currentZoneFrame, zone: this };
	                try {
	                    if (task.type == macroTask && task.data && !task.data.isPeriodic) {
	                        task.cancelFn = undefined;
	                    }
	                    try {
	                        return this._zoneDelegate.invokeTask(this, task, applyThis, applyArgs);
	                    }
	                    catch (error) {
	                        if (this._zoneDelegate.handleError(this, error)) {
	                            throw error;
	                        }
	                    }
	                }
	                finally {
	                    // if the task's state is notScheduled or unknown, then it has already been cancelled
	                    // we should not reset the state to scheduled
	                    if (task.state !== notScheduled && task.state !== unknown) {
	                        if (task.type == eventTask || (task.data && task.data.isPeriodic)) {
	                            reEntryGuard && task._transitionTo(scheduled, running);
	                        }
	                        else {
	                            task.runCount = 0;
	                            this._updateTaskCount(task, -1);
	                            reEntryGuard &&
	                                task._transitionTo(notScheduled, running, notScheduled);
	                        }
	                    }
	                    _currentZoneFrame = _currentZoneFrame.parent;
	                    _currentTask = previousTask;
	                }
	            };
	            Zone.prototype.scheduleTask = function (task) {
	                if (task.zone && task.zone !== this) {
	                    // check if the task was rescheduled, the newZone
	                    // should not be the children of the original zone
	                    var newZone = this;
	                    while (newZone) {
	                        if (newZone === task.zone) {
	                            throw Error("can not reschedule task to ".concat(this.name, " which is descendants of the original zone ").concat(task.zone.name));
	                        }
	                        newZone = newZone.parent;
	                    }
	                }
	                task._transitionTo(scheduling, notScheduled);
	                var zoneDelegates = [];
	                task._zoneDelegates = zoneDelegates;
	                task._zone = this;
	                try {
	                    task = this._zoneDelegate.scheduleTask(this, task);
	                }
	                catch (err) {
	                    // should set task's state to unknown when scheduleTask throw error
	                    // because the err may from reschedule, so the fromState maybe notScheduled
	                    task._transitionTo(unknown, scheduling, notScheduled);
	                    // TODO: @JiaLiPassion, should we check the result from handleError?
	                    this._zoneDelegate.handleError(this, err);
	                    throw err;
	                }
	                if (task._zoneDelegates === zoneDelegates) {
	                    // we have to check because internally the delegate can reschedule the task.
	                    this._updateTaskCount(task, 1);
	                }
	                if (task.state == scheduling) {
	                    task._transitionTo(scheduled, scheduling);
	                }
	                return task;
	            };
	            Zone.prototype.scheduleMicroTask = function (source, callback, data, customSchedule) {
	                return this.scheduleTask(new ZoneTask(microTask, source, callback, data, customSchedule, undefined));
	            };
	            Zone.prototype.scheduleMacroTask = function (source, callback, data, customSchedule, customCancel) {
	                return this.scheduleTask(new ZoneTask(macroTask, source, callback, data, customSchedule, customCancel));
	            };
	            Zone.prototype.scheduleEventTask = function (source, callback, data, customSchedule, customCancel) {
	                return this.scheduleTask(new ZoneTask(eventTask, source, callback, data, customSchedule, customCancel));
	            };
	            Zone.prototype.cancelTask = function (task) {
	                if (task.zone != this)
	                    throw new Error('A task can only be cancelled in the zone of creation! (Creation: ' +
	                        (task.zone || NO_ZONE).name + '; Execution: ' + this.name + ')');
	                if (task.state !== scheduled && task.state !== running) {
	                    return;
	                }
	                task._transitionTo(canceling, scheduled, running);
	                try {
	                    this._zoneDelegate.cancelTask(this, task);
	                }
	                catch (err) {
	                    // if error occurs when cancelTask, transit the state to unknown
	                    task._transitionTo(unknown, canceling);
	                    this._zoneDelegate.handleError(this, err);
	                    throw err;
	                }
	                this._updateTaskCount(task, -1);
	                task._transitionTo(notScheduled, canceling);
	                task.runCount = 0;
	                return task;
	            };
	            Zone.prototype._updateTaskCount = function (task, count) {
	                var zoneDelegates = task._zoneDelegates;
	                if (count == -1) {
	                    task._zoneDelegates = null;
	                }
	                for (var i = 0; i < zoneDelegates.length; i++) {
	                    zoneDelegates[i]._updateTaskCount(task.type, count);
	                }
	            };
	            return Zone;
	        }());
	        // tslint:disable-next-line:require-internal-with-underscore
	        Zone.__symbol__ = __symbol__;
	        var DELEGATE_ZS = {
	            name: '',
	            onHasTask: function (delegate, _, target, hasTaskState) { return delegate.hasTask(target, hasTaskState); },
	            onScheduleTask: function (delegate, _, target, task) { return delegate.scheduleTask(target, task); },
	            onInvokeTask: function (delegate, _, target, task, applyThis, applyArgs) { return delegate.invokeTask(target, task, applyThis, applyArgs); },
	            onCancelTask: function (delegate, _, target, task) { return delegate.cancelTask(target, task); }
	        };
	        var _ZoneDelegate = /** @class */ (function () {
	            function _ZoneDelegate(zone, parentDelegate, zoneSpec) {
	                this._taskCounts = { 'microTask': 0, 'macroTask': 0, 'eventTask': 0 };
	                this.zone = zone;
	                this._parentDelegate = parentDelegate;
	                this._forkZS = zoneSpec && (zoneSpec && zoneSpec.onFork ? zoneSpec : parentDelegate._forkZS);
	                this._forkDlgt = zoneSpec && (zoneSpec.onFork ? parentDelegate : parentDelegate._forkDlgt);
	                this._forkCurrZone =
	                    zoneSpec && (zoneSpec.onFork ? this.zone : parentDelegate._forkCurrZone);
	                this._interceptZS =
	                    zoneSpec && (zoneSpec.onIntercept ? zoneSpec : parentDelegate._interceptZS);
	                this._interceptDlgt =
	                    zoneSpec && (zoneSpec.onIntercept ? parentDelegate : parentDelegate._interceptDlgt);
	                this._interceptCurrZone =
	                    zoneSpec && (zoneSpec.onIntercept ? this.zone : parentDelegate._interceptCurrZone);
	                this._invokeZS = zoneSpec && (zoneSpec.onInvoke ? zoneSpec : parentDelegate._invokeZS);
	                this._invokeDlgt =
	                    zoneSpec && (zoneSpec.onInvoke ? parentDelegate : parentDelegate._invokeDlgt);
	                this._invokeCurrZone =
	                    zoneSpec && (zoneSpec.onInvoke ? this.zone : parentDelegate._invokeCurrZone);
	                this._handleErrorZS =
	                    zoneSpec && (zoneSpec.onHandleError ? zoneSpec : parentDelegate._handleErrorZS);
	                this._handleErrorDlgt =
	                    zoneSpec && (zoneSpec.onHandleError ? parentDelegate : parentDelegate._handleErrorDlgt);
	                this._handleErrorCurrZone =
	                    zoneSpec && (zoneSpec.onHandleError ? this.zone : parentDelegate._handleErrorCurrZone);
	                this._scheduleTaskZS =
	                    zoneSpec && (zoneSpec.onScheduleTask ? zoneSpec : parentDelegate._scheduleTaskZS);
	                this._scheduleTaskDlgt = zoneSpec &&
	                    (zoneSpec.onScheduleTask ? parentDelegate : parentDelegate._scheduleTaskDlgt);
	                this._scheduleTaskCurrZone =
	                    zoneSpec && (zoneSpec.onScheduleTask ? this.zone : parentDelegate._scheduleTaskCurrZone);
	                this._invokeTaskZS =
	                    zoneSpec && (zoneSpec.onInvokeTask ? zoneSpec : parentDelegate._invokeTaskZS);
	                this._invokeTaskDlgt =
	                    zoneSpec && (zoneSpec.onInvokeTask ? parentDelegate : parentDelegate._invokeTaskDlgt);
	                this._invokeTaskCurrZone =
	                    zoneSpec && (zoneSpec.onInvokeTask ? this.zone : parentDelegate._invokeTaskCurrZone);
	                this._cancelTaskZS =
	                    zoneSpec && (zoneSpec.onCancelTask ? zoneSpec : parentDelegate._cancelTaskZS);
	                this._cancelTaskDlgt =
	                    zoneSpec && (zoneSpec.onCancelTask ? parentDelegate : parentDelegate._cancelTaskDlgt);
	                this._cancelTaskCurrZone =
	                    zoneSpec && (zoneSpec.onCancelTask ? this.zone : parentDelegate._cancelTaskCurrZone);
	                this._hasTaskZS = null;
	                this._hasTaskDlgt = null;
	                this._hasTaskDlgtOwner = null;
	                this._hasTaskCurrZone = null;
	                var zoneSpecHasTask = zoneSpec && zoneSpec.onHasTask;
	                var parentHasTask = parentDelegate && parentDelegate._hasTaskZS;
	                if (zoneSpecHasTask || parentHasTask) {
	                    // If we need to report hasTask, than this ZS needs to do ref counting on tasks. In such
	                    // a case all task related interceptors must go through this ZD. We can't short circuit it.
	                    this._hasTaskZS = zoneSpecHasTask ? zoneSpec : DELEGATE_ZS;
	                    this._hasTaskDlgt = parentDelegate;
	                    this._hasTaskDlgtOwner = this;
	                    this._hasTaskCurrZone = zone;
	                    if (!zoneSpec.onScheduleTask) {
	                        this._scheduleTaskZS = DELEGATE_ZS;
	                        this._scheduleTaskDlgt = parentDelegate;
	                        this._scheduleTaskCurrZone = this.zone;
	                    }
	                    if (!zoneSpec.onInvokeTask) {
	                        this._invokeTaskZS = DELEGATE_ZS;
	                        this._invokeTaskDlgt = parentDelegate;
	                        this._invokeTaskCurrZone = this.zone;
	                    }
	                    if (!zoneSpec.onCancelTask) {
	                        this._cancelTaskZS = DELEGATE_ZS;
	                        this._cancelTaskDlgt = parentDelegate;
	                        this._cancelTaskCurrZone = this.zone;
	                    }
	                }
	            }
	            _ZoneDelegate.prototype.fork = function (targetZone, zoneSpec) {
	                return this._forkZS ? this._forkZS.onFork(this._forkDlgt, this.zone, targetZone, zoneSpec) :
	                    new Zone(targetZone, zoneSpec);
	            };
	            _ZoneDelegate.prototype.intercept = function (targetZone, callback, source) {
	                return this._interceptZS ?
	                    this._interceptZS.onIntercept(this._interceptDlgt, this._interceptCurrZone, targetZone, callback, source) :
	                    callback;
	            };
	            _ZoneDelegate.prototype.invoke = function (targetZone, callback, applyThis, applyArgs, source) {
	                return this._invokeZS ? this._invokeZS.onInvoke(this._invokeDlgt, this._invokeCurrZone, targetZone, callback, applyThis, applyArgs, source) :
	                    callback.apply(applyThis, applyArgs);
	            };
	            _ZoneDelegate.prototype.handleError = function (targetZone, error) {
	                return this._handleErrorZS ?
	                    this._handleErrorZS.onHandleError(this._handleErrorDlgt, this._handleErrorCurrZone, targetZone, error) :
	                    true;
	            };
	            _ZoneDelegate.prototype.scheduleTask = function (targetZone, task) {
	                var returnTask = task;
	                if (this._scheduleTaskZS) {
	                    if (this._hasTaskZS) {
	                        returnTask._zoneDelegates.push(this._hasTaskDlgtOwner);
	                    }
	                    // clang-format off
	                    returnTask = this._scheduleTaskZS.onScheduleTask(this._scheduleTaskDlgt, this._scheduleTaskCurrZone, targetZone, task);
	                    // clang-format on
	                    if (!returnTask)
	                        returnTask = task;
	                }
	                else {
	                    if (task.scheduleFn) {
	                        task.scheduleFn(task);
	                    }
	                    else if (task.type == microTask) {
	                        scheduleMicroTask(task);
	                    }
	                    else {
	                        throw new Error('Task is missing scheduleFn.');
	                    }
	                }
	                return returnTask;
	            };
	            _ZoneDelegate.prototype.invokeTask = function (targetZone, task, applyThis, applyArgs) {
	                return this._invokeTaskZS ? this._invokeTaskZS.onInvokeTask(this._invokeTaskDlgt, this._invokeTaskCurrZone, targetZone, task, applyThis, applyArgs) :
	                    task.callback.apply(applyThis, applyArgs);
	            };
	            _ZoneDelegate.prototype.cancelTask = function (targetZone, task) {
	                var value;
	                if (this._cancelTaskZS) {
	                    value = this._cancelTaskZS.onCancelTask(this._cancelTaskDlgt, this._cancelTaskCurrZone, targetZone, task);
	                }
	                else {
	                    if (!task.cancelFn) {
	                        throw Error('Task is not cancelable');
	                    }
	                    value = task.cancelFn(task);
	                }
	                return value;
	            };
	            _ZoneDelegate.prototype.hasTask = function (targetZone, isEmpty) {
	                // hasTask should not throw error so other ZoneDelegate
	                // can still trigger hasTask callback
	                try {
	                    this._hasTaskZS &&
	                        this._hasTaskZS.onHasTask(this._hasTaskDlgt, this._hasTaskCurrZone, targetZone, isEmpty);
	                }
	                catch (err) {
	                    this.handleError(targetZone, err);
	                }
	            };
	            // tslint:disable-next-line:require-internal-with-underscore
	            _ZoneDelegate.prototype._updateTaskCount = function (type, count) {
	                var counts = this._taskCounts;
	                var prev = counts[type];
	                var next = counts[type] = prev + count;
	                if (next < 0) {
	                    throw new Error('More tasks executed then were scheduled.');
	                }
	                if (prev == 0 || next == 0) {
	                    var isEmpty = {
	                        microTask: counts['microTask'] > 0,
	                        macroTask: counts['macroTask'] > 0,
	                        eventTask: counts['eventTask'] > 0,
	                        change: type
	                    };
	                    this.hasTask(this.zone, isEmpty);
	                }
	            };
	            return _ZoneDelegate;
	        }());
	        var ZoneTask = /** @class */ (function () {
	            function ZoneTask(type, source, callback, options, scheduleFn, cancelFn) {
	                // tslint:disable-next-line:require-internal-with-underscore
	                this._zone = null;
	                this.runCount = 0;
	                // tslint:disable-next-line:require-internal-with-underscore
	                this._zoneDelegates = null;
	                // tslint:disable-next-line:require-internal-with-underscore
	                this._state = 'notScheduled';
	                this.type = type;
	                this.source = source;
	                this.data = options;
	                this.scheduleFn = scheduleFn;
	                this.cancelFn = cancelFn;
	                if (!callback) {
	                    throw new Error('callback is not defined');
	                }
	                this.callback = callback;
	                var self = this;
	                // TODO: @JiaLiPassion options should have interface
	                if (type === eventTask && options && options.useG) {
	                    this.invoke = ZoneTask.invokeTask;
	                }
	                else {
	                    this.invoke = function () {
	                        return ZoneTask.invokeTask.call(global, self, this, arguments);
	                    };
	                }
	            }
	            ZoneTask.invokeTask = function (task, target, args) {
	                if (!task) {
	                    task = this;
	                }
	                _numberOfNestedTaskFrames++;
	                try {
	                    task.runCount++;
	                    return task.zone.runTask(task, target, args);
	                }
	                finally {
	                    if (_numberOfNestedTaskFrames == 1) {
	                        drainMicroTaskQueue();
	                    }
	                    _numberOfNestedTaskFrames--;
	                }
	            };
	            Object.defineProperty(ZoneTask.prototype, "zone", {
	                get: function () {
	                    return this._zone;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Object.defineProperty(ZoneTask.prototype, "state", {
	                get: function () {
	                    return this._state;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            ZoneTask.prototype.cancelScheduleRequest = function () {
	                this._transitionTo(notScheduled, scheduling);
	            };
	            // tslint:disable-next-line:require-internal-with-underscore
	            ZoneTask.prototype._transitionTo = function (toState, fromState1, fromState2) {
	                if (this._state === fromState1 || this._state === fromState2) {
	                    this._state = toState;
	                    if (toState == notScheduled) {
	                        this._zoneDelegates = null;
	                    }
	                }
	                else {
	                    throw new Error("".concat(this.type, " '").concat(this.source, "': can not transition to '").concat(toState, "', expecting state '").concat(fromState1, "'").concat(fromState2 ? ' or \'' + fromState2 + '\'' : '', ", was '").concat(this._state, "'."));
	                }
	            };
	            ZoneTask.prototype.toString = function () {
	                if (this.data && typeof this.data.handleId !== 'undefined') {
	                    return this.data.handleId.toString();
	                }
	                else {
	                    return Object.prototype.toString.call(this);
	                }
	            };
	            // add toJSON method to prevent cyclic error when
	            // call JSON.stringify(zoneTask)
	            ZoneTask.prototype.toJSON = function () {
	                return {
	                    type: this.type,
	                    state: this.state,
	                    source: this.source,
	                    zone: this.zone.name,
	                    runCount: this.runCount
	                };
	            };
	            return ZoneTask;
	        }());
	        //////////////////////////////////////////////////////
	        //////////////////////////////////////////////////////
	        ///  MICROTASK QUEUE
	        //////////////////////////////////////////////////////
	        //////////////////////////////////////////////////////
	        var symbolSetTimeout = __symbol__('setTimeout');
	        var symbolPromise = __symbol__('Promise');
	        var symbolThen = __symbol__('then');
	        var _microTaskQueue = [];
	        var _isDrainingMicrotaskQueue = false;
	        var nativeMicroTaskQueuePromise;
	        function nativeScheduleMicroTask(func) {
	            if (!nativeMicroTaskQueuePromise) {
	                if (global[symbolPromise]) {
	                    nativeMicroTaskQueuePromise = global[symbolPromise].resolve(0);
	                }
	            }
	            if (nativeMicroTaskQueuePromise) {
	                var nativeThen = nativeMicroTaskQueuePromise[symbolThen];
	                if (!nativeThen) {
	                    // native Promise is not patchable, we need to use `then` directly
	                    // issue 1078
	                    nativeThen = nativeMicroTaskQueuePromise['then'];
	                }
	                nativeThen.call(nativeMicroTaskQueuePromise, func);
	            }
	            else {
	                global[symbolSetTimeout](func, 0);
	            }
	        }
	        function scheduleMicroTask(task) {
	            // if we are not running in any task, and there has not been anything scheduled
	            // we must bootstrap the initial task creation by manually scheduling the drain
	            if (_numberOfNestedTaskFrames === 0 && _microTaskQueue.length === 0) {
	                // We are not running in Task, so we need to kickstart the microtask queue.
	                nativeScheduleMicroTask(drainMicroTaskQueue);
	            }
	            task && _microTaskQueue.push(task);
	        }
	        function drainMicroTaskQueue() {
	            if (!_isDrainingMicrotaskQueue) {
	                _isDrainingMicrotaskQueue = true;
	                while (_microTaskQueue.length) {
	                    var queue = _microTaskQueue;
	                    _microTaskQueue = [];
	                    for (var i = 0; i < queue.length; i++) {
	                        var task = queue[i];
	                        try {
	                            task.zone.runTask(task, null, null);
	                        }
	                        catch (error) {
	                            _api.onUnhandledError(error);
	                        }
	                    }
	                }
	                _api.microtaskDrainDone();
	                _isDrainingMicrotaskQueue = false;
	            }
	        }
	        //////////////////////////////////////////////////////
	        //////////////////////////////////////////////////////
	        ///  BOOTSTRAP
	        //////////////////////////////////////////////////////
	        //////////////////////////////////////////////////////
	        var NO_ZONE = { name: 'NO ZONE' };
	        var notScheduled = 'notScheduled', scheduling = 'scheduling', scheduled = 'scheduled', running = 'running', canceling = 'canceling', unknown = 'unknown';
	        var microTask = 'microTask', macroTask = 'macroTask', eventTask = 'eventTask';
	        var patches = {};
	        var _api = {
	            symbol: __symbol__,
	            currentZoneFrame: function () { return _currentZoneFrame; },
	            onUnhandledError: noop,
	            microtaskDrainDone: noop,
	            scheduleMicroTask: scheduleMicroTask,
	            showUncaughtError: function () { return !Zone[__symbol__('ignoreConsoleErrorUncaughtError')]; },
	            patchEventTarget: function () { return []; },
	            patchOnProperties: noop,
	            patchMethod: function () { return noop; },
	            bindArguments: function () { return []; },
	            patchThen: function () { return noop; },
	            patchMacroTask: function () { return noop; },
	            patchEventPrototype: function () { return noop; },
	            isIEOrEdge: function () { return false; },
	            getGlobalObjects: function () { return undefined; },
	            ObjectDefineProperty: function () { return noop; },
	            ObjectGetOwnPropertyDescriptor: function () { return undefined; },
	            ObjectCreate: function () { return undefined; },
	            ArraySlice: function () { return []; },
	            patchClass: function () { return noop; },
	            wrapWithCurrentZone: function () { return noop; },
	            filterProperties: function () { return []; },
	            attachOriginToPatched: function () { return noop; },
	            _redefineProperty: function () { return noop; },
	            patchCallbacks: function () { return noop; },
	            nativeScheduleMicroTask: nativeScheduleMicroTask
	        };
	        var _currentZoneFrame = { parent: null, zone: new Zone(null, null) };
	        var _currentTask = null;
	        var _numberOfNestedTaskFrames = 0;
	        function noop() { }
	        performanceMeasure('Zone', 'Zone');
	        return global['Zone'] = Zone;
	    }))(typeof window !== 'undefined' && window || typeof self !== 'undefined' && self || commonjsGlobal);
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    /**
	     * Suppress closure compiler errors about unknown 'Zone' variable
	     * @fileoverview
	     * @suppress {undefinedVars,globalThis,missingRequire}
	     */
	    /// <reference types="node"/>
	    // issue #989, to reduce bundle size, use short name
	    /** Object.getOwnPropertyDescriptor */
	    var ObjectGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
	    /** Object.defineProperty */
	    var ObjectDefineProperty = Object.defineProperty;
	    /** Object.getPrototypeOf */
	    var ObjectGetPrototypeOf = Object.getPrototypeOf;
	    /** Object.create */
	    var ObjectCreate = Object.create;
	    /** Array.prototype.slice */
	    var ArraySlice = Array.prototype.slice;
	    /** addEventListener string const */
	    var ADD_EVENT_LISTENER_STR = 'addEventListener';
	    /** removeEventListener string const */
	    var REMOVE_EVENT_LISTENER_STR = 'removeEventListener';
	    /** zoneSymbol addEventListener */
	    var ZONE_SYMBOL_ADD_EVENT_LISTENER = Zone.__symbol__(ADD_EVENT_LISTENER_STR);
	    /** zoneSymbol removeEventListener */
	    var ZONE_SYMBOL_REMOVE_EVENT_LISTENER = Zone.__symbol__(REMOVE_EVENT_LISTENER_STR);
	    /** true string const */
	    var TRUE_STR = 'true';
	    /** false string const */
	    var FALSE_STR = 'false';
	    /** Zone symbol prefix string const. */
	    var ZONE_SYMBOL_PREFIX = Zone.__symbol__('');
	    function wrapWithCurrentZone(callback, source) {
	        return Zone.current.wrap(callback, source);
	    }
	    function scheduleMacroTaskWithCurrentZone(source, callback, data, customSchedule, customCancel) {
	        return Zone.current.scheduleMacroTask(source, callback, data, customSchedule, customCancel);
	    }
	    var zoneSymbol$1 = Zone.__symbol__;
	    var isWindowExists = typeof window !== 'undefined';
	    var internalWindow = isWindowExists ? window : undefined;
	    var _global = isWindowExists && internalWindow || typeof self === 'object' && self || commonjsGlobal;
	    var REMOVE_ATTRIBUTE = 'removeAttribute';
	    function bindArguments(args, source) {
	        for (var i = args.length - 1; i >= 0; i--) {
	            if (typeof args[i] === 'function') {
	                args[i] = wrapWithCurrentZone(args[i], source + '_' + i);
	            }
	        }
	        return args;
	    }
	    function patchPrototype(prototype, fnNames) {
	        var source = prototype.constructor['name'];
	        var _loop_1 = function (i) {
	            var name_1 = fnNames[i];
	            var delegate = prototype[name_1];
	            if (delegate) {
	                var prototypeDesc = ObjectGetOwnPropertyDescriptor(prototype, name_1);
	                if (!isPropertyWritable(prototypeDesc)) {
	                    return "continue";
	                }
	                prototype[name_1] = (function (delegate) {
	                    var patched = function () {
	                        return delegate.apply(this, bindArguments(arguments, source + '.' + name_1));
	                    };
	                    attachOriginToPatched(patched, delegate);
	                    return patched;
	                })(delegate);
	            }
	        };
	        for (var i = 0; i < fnNames.length; i++) {
	            _loop_1(i);
	        }
	    }
	    function isPropertyWritable(propertyDesc) {
	        if (!propertyDesc) {
	            return true;
	        }
	        if (propertyDesc.writable === false) {
	            return false;
	        }
	        return !(typeof propertyDesc.get === 'function' && typeof propertyDesc.set === 'undefined');
	    }
	    var isWebWorker = (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope);
	    // Make sure to access `process` through `_global` so that WebPack does not accidentally browserify
	    // this code.
	    var isNode = (!('nw' in _global) && typeof _global.process !== 'undefined' &&
	        {}.toString.call(_global.process) === '[object process]');
	    var isBrowser = !isNode && !isWebWorker && !!(isWindowExists && internalWindow['HTMLElement']);
	    // we are in electron of nw, so we are both browser and nodejs
	    // Make sure to access `process` through `_global` so that WebPack does not accidentally browserify
	    // this code.
	    var isMix = typeof _global.process !== 'undefined' &&
	        {}.toString.call(_global.process) === '[object process]' && !isWebWorker &&
	        !!(isWindowExists && internalWindow['HTMLElement']);
	    var zoneSymbolEventNames$1 = {};
	    var wrapFn = function (event) {
	        // https://github.com/angular/zone.js/issues/911, in IE, sometimes
	        // event will be undefined, so we need to use window.event
	        event = event || _global.event;
	        if (!event) {
	            return;
	        }
	        var eventNameSymbol = zoneSymbolEventNames$1[event.type];
	        if (!eventNameSymbol) {
	            eventNameSymbol = zoneSymbolEventNames$1[event.type] = zoneSymbol$1('ON_PROPERTY' + event.type);
	        }
	        var target = this || event.target || _global;
	        var listener = target[eventNameSymbol];
	        var result;
	        if (isBrowser && target === internalWindow && event.type === 'error') {
	            // window.onerror have different signature
	            // https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onerror#window.onerror
	            // and onerror callback will prevent default when callback return true
	            var errorEvent = event;
	            result = listener &&
	                listener.call(this, errorEvent.message, errorEvent.filename, errorEvent.lineno, errorEvent.colno, errorEvent.error);
	            if (result === true) {
	                event.preventDefault();
	            }
	        }
	        else {
	            result = listener && listener.apply(this, arguments);
	            if (result != undefined && !result) {
	                event.preventDefault();
	            }
	        }
	        return result;
	    };
	    function patchProperty(obj, prop, prototype) {
	        var desc = ObjectGetOwnPropertyDescriptor(obj, prop);
	        if (!desc && prototype) {
	            // when patch window object, use prototype to check prop exist or not
	            var prototypeDesc = ObjectGetOwnPropertyDescriptor(prototype, prop);
	            if (prototypeDesc) {
	                desc = { enumerable: true, configurable: true };
	            }
	        }
	        // if the descriptor not exists or is not configurable
	        // just return
	        if (!desc || !desc.configurable) {
	            return;
	        }
	        var onPropPatchedSymbol = zoneSymbol$1('on' + prop + 'patched');
	        if (obj.hasOwnProperty(onPropPatchedSymbol) && obj[onPropPatchedSymbol]) {
	            return;
	        }
	        // A property descriptor cannot have getter/setter and be writable
	        // deleting the writable and value properties avoids this error:
	        //
	        // TypeError: property descriptors must not specify a value or be writable when a
	        // getter or setter has been specified
	        delete desc.writable;
	        delete desc.value;
	        var originalDescGet = desc.get;
	        var originalDescSet = desc.set;
	        // slice(2) cuz 'onclick' -> 'click', etc
	        var eventName = prop.slice(2);
	        var eventNameSymbol = zoneSymbolEventNames$1[eventName];
	        if (!eventNameSymbol) {
	            eventNameSymbol = zoneSymbolEventNames$1[eventName] = zoneSymbol$1('ON_PROPERTY' + eventName);
	        }
	        desc.set = function (newValue) {
	            // in some of windows's onproperty callback, this is undefined
	            // so we need to check it
	            var target = this;
	            if (!target && obj === _global) {
	                target = _global;
	            }
	            if (!target) {
	                return;
	            }
	            var previousValue = target[eventNameSymbol];
	            if (typeof previousValue === 'function') {
	                target.removeEventListener(eventName, wrapFn);
	            }
	            // issue #978, when onload handler was added before loading zone.js
	            // we should remove it with originalDescSet
	            originalDescSet && originalDescSet.call(target, null);
	            target[eventNameSymbol] = newValue;
	            if (typeof newValue === 'function') {
	                target.addEventListener(eventName, wrapFn, false);
	            }
	        };
	        // The getter would return undefined for unassigned properties but the default value of an
	        // unassigned property is null
	        desc.get = function () {
	            // in some of windows's onproperty callback, this is undefined
	            // so we need to check it
	            var target = this;
	            if (!target && obj === _global) {
	                target = _global;
	            }
	            if (!target) {
	                return null;
	            }
	            var listener = target[eventNameSymbol];
	            if (listener) {
	                return listener;
	            }
	            else if (originalDescGet) {
	                // result will be null when use inline event attribute,
	                // such as <button onclick="func();">OK</button>
	                // because the onclick function is internal raw uncompiled handler
	                // the onclick will be evaluated when first time event was triggered or
	                // the property is accessed, https://github.com/angular/zone.js/issues/525
	                // so we should use original native get to retrieve the handler
	                var value = originalDescGet.call(this);
	                if (value) {
	                    desc.set.call(this, value);
	                    if (typeof target[REMOVE_ATTRIBUTE] === 'function') {
	                        target.removeAttribute(prop);
	                    }
	                    return value;
	                }
	            }
	            return null;
	        };
	        ObjectDefineProperty(obj, prop, desc);
	        obj[onPropPatchedSymbol] = true;
	    }
	    function patchOnProperties(obj, properties, prototype) {
	        if (properties) {
	            for (var i = 0; i < properties.length; i++) {
	                patchProperty(obj, 'on' + properties[i], prototype);
	            }
	        }
	        else {
	            var onProperties = [];
	            for (var prop in obj) {
	                if (prop.slice(0, 2) == 'on') {
	                    onProperties.push(prop);
	                }
	            }
	            for (var j = 0; j < onProperties.length; j++) {
	                patchProperty(obj, onProperties[j], prototype);
	            }
	        }
	    }
	    var originalInstanceKey = zoneSymbol$1('originalInstance');
	    // wrap some native API on `window`
	    function patchClass(className) {
	        var OriginalClass = _global[className];
	        if (!OriginalClass)
	            return;
	        // keep original class in global
	        _global[zoneSymbol$1(className)] = OriginalClass;
	        _global[className] = function () {
	            var a = bindArguments(arguments, className);
	            switch (a.length) {
	                case 0:
	                    this[originalInstanceKey] = new OriginalClass();
	                    break;
	                case 1:
	                    this[originalInstanceKey] = new OriginalClass(a[0]);
	                    break;
	                case 2:
	                    this[originalInstanceKey] = new OriginalClass(a[0], a[1]);
	                    break;
	                case 3:
	                    this[originalInstanceKey] = new OriginalClass(a[0], a[1], a[2]);
	                    break;
	                case 4:
	                    this[originalInstanceKey] = new OriginalClass(a[0], a[1], a[2], a[3]);
	                    break;
	                default:
	                    throw new Error('Arg list too long.');
	            }
	        };
	        // attach original delegate to patched function
	        attachOriginToPatched(_global[className], OriginalClass);
	        var instance = new OriginalClass(function () { });
	        var prop;
	        for (prop in instance) {
	            // https://bugs.webkit.org/show_bug.cgi?id=44721
	            if (className === 'XMLHttpRequest' && prop === 'responseBlob')
	                continue;
	            (function (prop) {
	                if (typeof instance[prop] === 'function') {
	                    _global[className].prototype[prop] = function () {
	                        return this[originalInstanceKey][prop].apply(this[originalInstanceKey], arguments);
	                    };
	                }
	                else {
	                    ObjectDefineProperty(_global[className].prototype, prop, {
	                        set: function (fn) {
	                            if (typeof fn === 'function') {
	                                this[originalInstanceKey][prop] = wrapWithCurrentZone(fn, className + '.' + prop);
	                                // keep callback in wrapped function so we can
	                                // use it in Function.prototype.toString to return
	                                // the native one.
	                                attachOriginToPatched(this[originalInstanceKey][prop], fn);
	                            }
	                            else {
	                                this[originalInstanceKey][prop] = fn;
	                            }
	                        },
	                        get: function () {
	                            return this[originalInstanceKey][prop];
	                        }
	                    });
	                }
	            }(prop));
	        }
	        for (prop in OriginalClass) {
	            if (prop !== 'prototype' && OriginalClass.hasOwnProperty(prop)) {
	                _global[className][prop] = OriginalClass[prop];
	            }
	        }
	    }
	    function patchMethod(target, name, patchFn) {
	        var proto = target;
	        while (proto && !proto.hasOwnProperty(name)) {
	            proto = ObjectGetPrototypeOf(proto);
	        }
	        if (!proto && target[name]) {
	            // somehow we did not find it, but we can see it. This happens on IE for Window properties.
	            proto = target;
	        }
	        var delegateName = zoneSymbol$1(name);
	        var delegate = null;
	        if (proto && (!(delegate = proto[delegateName]) || !proto.hasOwnProperty(delegateName))) {
	            delegate = proto[delegateName] = proto[name];
	            // check whether proto[name] is writable
	            // some property is readonly in safari, such as HtmlCanvasElement.prototype.toBlob
	            var desc = proto && ObjectGetOwnPropertyDescriptor(proto, name);
	            if (isPropertyWritable(desc)) {
	                var patchDelegate_1 = patchFn(delegate, delegateName, name);
	                proto[name] = function () {
	                    return patchDelegate_1(this, arguments);
	                };
	                attachOriginToPatched(proto[name], delegate);
	            }
	        }
	        return delegate;
	    }
	    // TODO: @JiaLiPassion, support cancel task later if necessary
	    function patchMacroTask(obj, funcName, metaCreator) {
	        var setNative = null;
	        function scheduleTask(task) {
	            var data = task.data;
	            data.args[data.cbIdx] = function () {
	                task.invoke.apply(this, arguments);
	            };
	            setNative.apply(data.target, data.args);
	            return task;
	        }
	        setNative = patchMethod(obj, funcName, function (delegate) { return function (self, args) {
	            var meta = metaCreator(self, args);
	            if (meta.cbIdx >= 0 && typeof args[meta.cbIdx] === 'function') {
	                return scheduleMacroTaskWithCurrentZone(meta.name, args[meta.cbIdx], meta, scheduleTask);
	            }
	            else {
	                // cause an error by calling it directly.
	                return delegate.apply(self, args);
	            }
	        }; });
	    }
	    function attachOriginToPatched(patched, original) {
	        patched[zoneSymbol$1('OriginalDelegate')] = original;
	    }
	    var isDetectedIEOrEdge = false;
	    var ieOrEdge = false;
	    function isIE() {
	        try {
	            var ua = internalWindow.navigator.userAgent;
	            if (ua.indexOf('MSIE ') !== -1 || ua.indexOf('Trident/') !== -1) {
	                return true;
	            }
	        }
	        catch (error) {
	        }
	        return false;
	    }
	    function isIEOrEdge() {
	        if (isDetectedIEOrEdge) {
	            return ieOrEdge;
	        }
	        isDetectedIEOrEdge = true;
	        try {
	            var ua = internalWindow.navigator.userAgent;
	            if (ua.indexOf('MSIE ') !== -1 || ua.indexOf('Trident/') !== -1 || ua.indexOf('Edge/') !== -1) {
	                ieOrEdge = true;
	            }
	        }
	        catch (error) {
	        }
	        return ieOrEdge;
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    Zone.__load_patch('ZoneAwarePromise', function (global, Zone, api) {
	        var ObjectGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
	        var ObjectDefineProperty = Object.defineProperty;
	        function readableObjectToString(obj) {
	            if (obj && obj.toString === Object.prototype.toString) {
	                var className = obj.constructor && obj.constructor.name;
	                return (className ? className : '') + ': ' + JSON.stringify(obj);
	            }
	            return obj ? obj.toString() : Object.prototype.toString.call(obj);
	        }
	        var __symbol__ = api.symbol;
	        var _uncaughtPromiseErrors = [];
	        var isDisableWrappingUncaughtPromiseRejection = global[__symbol__('DISABLE_WRAPPING_UNCAUGHT_PROMISE_REJECTION')] === true;
	        var symbolPromise = __symbol__('Promise');
	        var symbolThen = __symbol__('then');
	        var creationTrace = '__creationTrace__';
	        api.onUnhandledError = function (e) {
	            if (api.showUncaughtError()) {
	                var rejection = e && e.rejection;
	                if (rejection) {
	                    console.error('Unhandled Promise rejection:', rejection instanceof Error ? rejection.message : rejection, '; Zone:', e.zone.name, '; Task:', e.task && e.task.source, '; Value:', rejection, rejection instanceof Error ? rejection.stack : undefined);
	                }
	                else {
	                    console.error(e);
	                }
	            }
	        };
	        api.microtaskDrainDone = function () {
	            var _loop_2 = function () {
	                var uncaughtPromiseError = _uncaughtPromiseErrors.shift();
	                try {
	                    uncaughtPromiseError.zone.runGuarded(function () {
	                        if (uncaughtPromiseError.throwOriginal) {
	                            throw uncaughtPromiseError.rejection;
	                        }
	                        throw uncaughtPromiseError;
	                    });
	                }
	                catch (error) {
	                    handleUnhandledRejection(error);
	                }
	            };
	            while (_uncaughtPromiseErrors.length) {
	                _loop_2();
	            }
	        };
	        var UNHANDLED_PROMISE_REJECTION_HANDLER_SYMBOL = __symbol__('unhandledPromiseRejectionHandler');
	        function handleUnhandledRejection(e) {
	            api.onUnhandledError(e);
	            try {
	                var handler = Zone[UNHANDLED_PROMISE_REJECTION_HANDLER_SYMBOL];
	                if (typeof handler === 'function') {
	                    handler.call(this, e);
	                }
	            }
	            catch (err) {
	            }
	        }
	        function isThenable(value) {
	            return value && value.then;
	        }
	        function forwardResolution(value) {
	            return value;
	        }
	        function forwardRejection(rejection) {
	            return ZoneAwarePromise.reject(rejection);
	        }
	        var symbolState = __symbol__('state');
	        var symbolValue = __symbol__('value');
	        var symbolFinally = __symbol__('finally');
	        var symbolParentPromiseValue = __symbol__('parentPromiseValue');
	        var symbolParentPromiseState = __symbol__('parentPromiseState');
	        var source = 'Promise.then';
	        var UNRESOLVED = null;
	        var RESOLVED = true;
	        var REJECTED = false;
	        var REJECTED_NO_CATCH = 0;
	        function makeResolver(promise, state) {
	            return function (v) {
	                try {
	                    resolvePromise(promise, state, v);
	                }
	                catch (err) {
	                    resolvePromise(promise, false, err);
	                }
	                // Do not return value or you will break the Promise spec.
	            };
	        }
	        var once = function () {
	            var wasCalled = false;
	            return function wrapper(wrappedFunction) {
	                return function () {
	                    if (wasCalled) {
	                        return;
	                    }
	                    wasCalled = true;
	                    wrappedFunction.apply(null, arguments);
	                };
	            };
	        };
	        var TYPE_ERROR = 'Promise resolved with itself';
	        var CURRENT_TASK_TRACE_SYMBOL = __symbol__('currentTaskTrace');
	        // Promise Resolution
	        function resolvePromise(promise, state, value) {
	            var onceWrapper = once();
	            if (promise === value) {
	                throw new TypeError(TYPE_ERROR);
	            }
	            if (promise[symbolState] === UNRESOLVED) {
	                // should only get value.then once based on promise spec.
	                var then = null;
	                try {
	                    if (typeof value === 'object' || typeof value === 'function') {
	                        then = value && value.then;
	                    }
	                }
	                catch (err) {
	                    onceWrapper(function () {
	                        resolvePromise(promise, false, err);
	                    })();
	                    return promise;
	                }
	                // if (value instanceof ZoneAwarePromise) {
	                if (state !== REJECTED && value instanceof ZoneAwarePromise &&
	                    value.hasOwnProperty(symbolState) && value.hasOwnProperty(symbolValue) &&
	                    value[symbolState] !== UNRESOLVED) {
	                    clearRejectedNoCatch(value);
	                    resolvePromise(promise, value[symbolState], value[symbolValue]);
	                }
	                else if (state !== REJECTED && typeof then === 'function') {
	                    try {
	                        then.call(value, onceWrapper(makeResolver(promise, state)), onceWrapper(makeResolver(promise, false)));
	                    }
	                    catch (err) {
	                        onceWrapper(function () {
	                            resolvePromise(promise, false, err);
	                        })();
	                    }
	                }
	                else {
	                    promise[symbolState] = state;
	                    var queue = promise[symbolValue];
	                    promise[symbolValue] = value;
	                    if (promise[symbolFinally] === symbolFinally) {
	                        // the promise is generated by Promise.prototype.finally
	                        if (state === RESOLVED) {
	                            // the state is resolved, should ignore the value
	                            // and use parent promise value
	                            promise[symbolState] = promise[symbolParentPromiseState];
	                            promise[symbolValue] = promise[symbolParentPromiseValue];
	                        }
	                    }
	                    // record task information in value when error occurs, so we can
	                    // do some additional work such as render longStackTrace
	                    if (state === REJECTED && value instanceof Error) {
	                        // check if longStackTraceZone is here
	                        var trace = Zone.currentTask && Zone.currentTask.data &&
	                            Zone.currentTask.data[creationTrace];
	                        if (trace) {
	                            // only keep the long stack trace into error when in longStackTraceZone
	                            ObjectDefineProperty(value, CURRENT_TASK_TRACE_SYMBOL, { configurable: true, enumerable: false, writable: true, value: trace });
	                        }
	                    }
	                    for (var i = 0; i < queue.length;) {
	                        scheduleResolveOrReject(promise, queue[i++], queue[i++], queue[i++], queue[i++]);
	                    }
	                    if (queue.length == 0 && state == REJECTED) {
	                        promise[symbolState] = REJECTED_NO_CATCH;
	                        var uncaughtPromiseError = value;
	                        try {
	                            // Here we throws a new Error to print more readable error log
	                            // and if the value is not an error, zone.js builds an `Error`
	                            // Object here to attach the stack information.
	                            throw new Error('Uncaught (in promise): ' + readableObjectToString(value) +
	                                (value && value.stack ? '\n' + value.stack : ''));
	                        }
	                        catch (err) {
	                            uncaughtPromiseError = err;
	                        }
	                        if (isDisableWrappingUncaughtPromiseRejection) {
	                            // If disable wrapping uncaught promise reject
	                            // use the value instead of wrapping it.
	                            uncaughtPromiseError.throwOriginal = true;
	                        }
	                        uncaughtPromiseError.rejection = value;
	                        uncaughtPromiseError.promise = promise;
	                        uncaughtPromiseError.zone = Zone.current;
	                        uncaughtPromiseError.task = Zone.currentTask;
	                        _uncaughtPromiseErrors.push(uncaughtPromiseError);
	                        api.scheduleMicroTask(); // to make sure that it is running
	                    }
	                }
	            }
	            // Resolving an already resolved promise is a noop.
	            return promise;
	        }
	        var REJECTION_HANDLED_HANDLER = __symbol__('rejectionHandledHandler');
	        function clearRejectedNoCatch(promise) {
	            if (promise[symbolState] === REJECTED_NO_CATCH) {
	                // if the promise is rejected no catch status
	                // and queue.length > 0, means there is a error handler
	                // here to handle the rejected promise, we should trigger
	                // windows.rejectionhandled eventHandler or nodejs rejectionHandled
	                // eventHandler
	                try {
	                    var handler = Zone[REJECTION_HANDLED_HANDLER];
	                    if (handler && typeof handler === 'function') {
	                        handler.call(this, { rejection: promise[symbolValue], promise: promise });
	                    }
	                }
	                catch (err) {
	                }
	                promise[symbolState] = REJECTED;
	                for (var i = 0; i < _uncaughtPromiseErrors.length; i++) {
	                    if (promise === _uncaughtPromiseErrors[i].promise) {
	                        _uncaughtPromiseErrors.splice(i, 1);
	                    }
	                }
	            }
	        }
	        function scheduleResolveOrReject(promise, zone, chainPromise, onFulfilled, onRejected) {
	            clearRejectedNoCatch(promise);
	            var promiseState = promise[symbolState];
	            var delegate = promiseState ?
	                (typeof onFulfilled === 'function') ? onFulfilled : forwardResolution :
	                (typeof onRejected === 'function') ? onRejected :
	                    forwardRejection;
	            zone.scheduleMicroTask(source, function () {
	                try {
	                    var parentPromiseValue = promise[symbolValue];
	                    var isFinallyPromise = !!chainPromise && symbolFinally === chainPromise[symbolFinally];
	                    if (isFinallyPromise) {
	                        // if the promise is generated from finally call, keep parent promise's state and value
	                        chainPromise[symbolParentPromiseValue] = parentPromiseValue;
	                        chainPromise[symbolParentPromiseState] = promiseState;
	                    }
	                    // should not pass value to finally callback
	                    var value = zone.run(delegate, undefined, isFinallyPromise && delegate !== forwardRejection && delegate !== forwardResolution ?
	                        [] :
	                        [parentPromiseValue]);
	                    resolvePromise(chainPromise, true, value);
	                }
	                catch (error) {
	                    // if error occurs, should always return this error
	                    resolvePromise(chainPromise, false, error);
	                }
	            }, chainPromise);
	        }
	        var ZONE_AWARE_PROMISE_TO_STRING = 'function ZoneAwarePromise() { [native code] }';
	        var noop = function () { };
	        var AggregateError = global.AggregateError;
	        var ZoneAwarePromise = /** @class */ (function () {
	            function ZoneAwarePromise(executor) {
	                var promise = this;
	                if (!(promise instanceof ZoneAwarePromise)) {
	                    throw new Error('Must be an instanceof Promise.');
	                }
	                promise[symbolState] = UNRESOLVED;
	                promise[symbolValue] = []; // queue;
	                try {
	                    var onceWrapper = once();
	                    executor &&
	                        executor(onceWrapper(makeResolver(promise, RESOLVED)), onceWrapper(makeResolver(promise, REJECTED)));
	                }
	                catch (error) {
	                    resolvePromise(promise, false, error);
	                }
	            }
	            ZoneAwarePromise.toString = function () {
	                return ZONE_AWARE_PROMISE_TO_STRING;
	            };
	            ZoneAwarePromise.resolve = function (value) {
	                return resolvePromise(new this(null), RESOLVED, value);
	            };
	            ZoneAwarePromise.reject = function (error) {
	                return resolvePromise(new this(null), REJECTED, error);
	            };
	            ZoneAwarePromise.any = function (values) {
	                if (!values || typeof values[Symbol.iterator] !== 'function') {
	                    return Promise.reject(new AggregateError([], 'All promises were rejected'));
	                }
	                var promises = [];
	                var count = 0;
	                try {
	                    for (var _i = 0, values_1 = values; _i < values_1.length; _i++) {
	                        var v = values_1[_i];
	                        count++;
	                        promises.push(ZoneAwarePromise.resolve(v));
	                    }
	                }
	                catch (err) {
	                    return Promise.reject(new AggregateError([], 'All promises were rejected'));
	                }
	                if (count === 0) {
	                    return Promise.reject(new AggregateError([], 'All promises were rejected'));
	                }
	                var finished = false;
	                var errors = [];
	                return new ZoneAwarePromise(function (resolve, reject) {
	                    for (var i = 0; i < promises.length; i++) {
	                        promises[i].then(function (v) {
	                            if (finished) {
	                                return;
	                            }
	                            finished = true;
	                            resolve(v);
	                        }, function (err) {
	                            errors.push(err);
	                            count--;
	                            if (count === 0) {
	                                finished = true;
	                                reject(new AggregateError(errors, 'All promises were rejected'));
	                            }
	                        });
	                    }
	                });
	            };
	            ZoneAwarePromise.race = function (values) {
	                var resolve;
	                var reject;
	                var promise = new this(function (res, rej) {
	                    resolve = res;
	                    reject = rej;
	                });
	                function onResolve(value) {
	                    resolve(value);
	                }
	                function onReject(error) {
	                    reject(error);
	                }
	                for (var _i = 0, values_2 = values; _i < values_2.length; _i++) {
	                    var value = values_2[_i];
	                    if (!isThenable(value)) {
	                        value = this.resolve(value);
	                    }
	                    value.then(onResolve, onReject);
	                }
	                return promise;
	            };
	            ZoneAwarePromise.all = function (values) {
	                return ZoneAwarePromise.allWithCallback(values);
	            };
	            ZoneAwarePromise.allSettled = function (values) {
	                var P = this && this.prototype instanceof ZoneAwarePromise ? this : ZoneAwarePromise;
	                return P.allWithCallback(values, {
	                    thenCallback: function (value) { return ({ status: 'fulfilled', value: value }); },
	                    errorCallback: function (err) { return ({ status: 'rejected', reason: err }); }
	                });
	            };
	            ZoneAwarePromise.allWithCallback = function (values, callback) {
	                var resolve;
	                var reject;
	                var promise = new this(function (res, rej) {
	                    resolve = res;
	                    reject = rej;
	                });
	                // Start at 2 to prevent prematurely resolving if .then is called immediately.
	                var unresolvedCount = 2;
	                var valueIndex = 0;
	                var resolvedValues = [];
	                var _loop_3 = function (value) {
	                    if (!isThenable(value)) {
	                        value = this_1.resolve(value);
	                    }
	                    var curValueIndex = valueIndex;
	                    try {
	                        value.then(function (value) {
	                            resolvedValues[curValueIndex] = callback ? callback.thenCallback(value) : value;
	                            unresolvedCount--;
	                            if (unresolvedCount === 0) {
	                                resolve(resolvedValues);
	                            }
	                        }, function (err) {
	                            if (!callback) {
	                                reject(err);
	                            }
	                            else {
	                                resolvedValues[curValueIndex] = callback.errorCallback(err);
	                                unresolvedCount--;
	                                if (unresolvedCount === 0) {
	                                    resolve(resolvedValues);
	                                }
	                            }
	                        });
	                    }
	                    catch (thenErr) {
	                        reject(thenErr);
	                    }
	                    unresolvedCount++;
	                    valueIndex++;
	                };
	                var this_1 = this;
	                for (var _i = 0, values_3 = values; _i < values_3.length; _i++) {
	                    var value = values_3[_i];
	                    _loop_3(value);
	                }
	                // Make the unresolvedCount zero-based again.
	                unresolvedCount -= 2;
	                if (unresolvedCount === 0) {
	                    resolve(resolvedValues);
	                }
	                return promise;
	            };
	            Object.defineProperty(ZoneAwarePromise.prototype, Symbol.toStringTag, {
	                get: function () {
	                    return 'Promise';
	                },
	                enumerable: false,
	                configurable: true
	            });
	            Object.defineProperty(ZoneAwarePromise.prototype, Symbol.species, {
	                get: function () {
	                    return ZoneAwarePromise;
	                },
	                enumerable: false,
	                configurable: true
	            });
	            ZoneAwarePromise.prototype.then = function (onFulfilled, onRejected) {
	                var _a;
	                // We must read `Symbol.species` safely because `this` may be anything. For instance, `this`
	                // may be an object without a prototype (created through `Object.create(null)`); thus
	                // `this.constructor` will be undefined. One of the use cases is SystemJS creating
	                // prototype-less objects (modules) via `Object.create(null)`. The SystemJS creates an empty
	                // object and copies promise properties into that object (within the `getOrCreateLoad`
	                // function). The zone.js then checks if the resolved value has the `then` method and invokes
	                // it with the `value` context. Otherwise, this will throw an error: `TypeError: Cannot read
	                // properties of undefined (reading 'Symbol(Symbol.species)')`.
	                var C = (_a = this.constructor) === null || _a === void 0 ? void 0 : _a[Symbol.species];
	                if (!C || typeof C !== 'function') {
	                    C = this.constructor || ZoneAwarePromise;
	                }
	                var chainPromise = new C(noop);
	                var zone = Zone.current;
	                if (this[symbolState] == UNRESOLVED) {
	                    this[symbolValue].push(zone, chainPromise, onFulfilled, onRejected);
	                }
	                else {
	                    scheduleResolveOrReject(this, zone, chainPromise, onFulfilled, onRejected);
	                }
	                return chainPromise;
	            };
	            ZoneAwarePromise.prototype.catch = function (onRejected) {
	                return this.then(null, onRejected);
	            };
	            ZoneAwarePromise.prototype.finally = function (onFinally) {
	                var _a;
	                // See comment on the call to `then` about why thee `Symbol.species` is safely accessed.
	                var C = (_a = this.constructor) === null || _a === void 0 ? void 0 : _a[Symbol.species];
	                if (!C || typeof C !== 'function') {
	                    C = ZoneAwarePromise;
	                }
	                var chainPromise = new C(noop);
	                chainPromise[symbolFinally] = symbolFinally;
	                var zone = Zone.current;
	                if (this[symbolState] == UNRESOLVED) {
	                    this[symbolValue].push(zone, chainPromise, onFinally, onFinally);
	                }
	                else {
	                    scheduleResolveOrReject(this, zone, chainPromise, onFinally, onFinally);
	                }
	                return chainPromise;
	            };
	            return ZoneAwarePromise;
	        }());
	        // Protect against aggressive optimizers dropping seemingly unused properties.
	        // E.g. Closure Compiler in advanced mode.
	        ZoneAwarePromise['resolve'] = ZoneAwarePromise.resolve;
	        ZoneAwarePromise['reject'] = ZoneAwarePromise.reject;
	        ZoneAwarePromise['race'] = ZoneAwarePromise.race;
	        ZoneAwarePromise['all'] = ZoneAwarePromise.all;
	        var NativePromise = global[symbolPromise] = global['Promise'];
	        global['Promise'] = ZoneAwarePromise;
	        var symbolThenPatched = __symbol__('thenPatched');
	        function patchThen(Ctor) {
	            var proto = Ctor.prototype;
	            var prop = ObjectGetOwnPropertyDescriptor(proto, 'then');
	            if (prop && (prop.writable === false || !prop.configurable)) {
	                // check Ctor.prototype.then propertyDescriptor is writable or not
	                // in meteor env, writable is false, we should ignore such case
	                return;
	            }
	            var originalThen = proto.then;
	            // Keep a reference to the original method.
	            proto[symbolThen] = originalThen;
	            Ctor.prototype.then = function (onResolve, onReject) {
	                var _this = this;
	                var wrapped = new ZoneAwarePromise(function (resolve, reject) {
	                    originalThen.call(_this, resolve, reject);
	                });
	                return wrapped.then(onResolve, onReject);
	            };
	            Ctor[symbolThenPatched] = true;
	        }
	        api.patchThen = patchThen;
	        function zoneify(fn) {
	            return function (self, args) {
	                var resultPromise = fn.apply(self, args);
	                if (resultPromise instanceof ZoneAwarePromise) {
	                    return resultPromise;
	                }
	                var ctor = resultPromise.constructor;
	                if (!ctor[symbolThenPatched]) {
	                    patchThen(ctor);
	                }
	                return resultPromise;
	            };
	        }
	        if (NativePromise) {
	            patchThen(NativePromise);
	            patchMethod(global, 'fetch', function (delegate) { return zoneify(delegate); });
	        }
	        // This is not part of public API, but it is useful for tests, so we expose it.
	        Promise[Zone.__symbol__('uncaughtPromiseErrors')] = _uncaughtPromiseErrors;
	        return ZoneAwarePromise;
	    });
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    // override Function.prototype.toString to make zone.js patched function
	    // look like native function
	    Zone.__load_patch('toString', function (global) {
	        // patch Func.prototype.toString to let them look like native
	        var originalFunctionToString = Function.prototype.toString;
	        var ORIGINAL_DELEGATE_SYMBOL = zoneSymbol$1('OriginalDelegate');
	        var PROMISE_SYMBOL = zoneSymbol$1('Promise');
	        var ERROR_SYMBOL = zoneSymbol$1('Error');
	        var newFunctionToString = function toString() {
	            if (typeof this === 'function') {
	                var originalDelegate = this[ORIGINAL_DELEGATE_SYMBOL];
	                if (originalDelegate) {
	                    if (typeof originalDelegate === 'function') {
	                        return originalFunctionToString.call(originalDelegate);
	                    }
	                    else {
	                        return Object.prototype.toString.call(originalDelegate);
	                    }
	                }
	                if (this === Promise) {
	                    var nativePromise = global[PROMISE_SYMBOL];
	                    if (nativePromise) {
	                        return originalFunctionToString.call(nativePromise);
	                    }
	                }
	                if (this === Error) {
	                    var nativeError = global[ERROR_SYMBOL];
	                    if (nativeError) {
	                        return originalFunctionToString.call(nativeError);
	                    }
	                }
	            }
	            return originalFunctionToString.call(this);
	        };
	        newFunctionToString[ORIGINAL_DELEGATE_SYMBOL] = originalFunctionToString;
	        Function.prototype.toString = newFunctionToString;
	        // patch Object.prototype.toString to let them look like native
	        var originalObjectToString = Object.prototype.toString;
	        var PROMISE_OBJECT_TO_STRING = '[object Promise]';
	        Object.prototype.toString = function () {
	            if (typeof Promise === 'function' && this instanceof Promise) {
	                return PROMISE_OBJECT_TO_STRING;
	            }
	            return originalObjectToString.call(this);
	        };
	    });
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    var passiveSupported = false;
	    if (typeof window !== 'undefined') {
	        try {
	            var options = Object.defineProperty({}, 'passive', {
	                get: function () {
	                    passiveSupported = true;
	                }
	            });
	            // Note: We pass the `options` object as the event handler too. This is not compatible with the
	            // signature of `addEventListener` or `removeEventListener` but enables us to remove the handler
	            // without an actual handler.
	            window.addEventListener('test', options, options);
	            window.removeEventListener('test', options, options);
	        }
	        catch (err) {
	            passiveSupported = false;
	        }
	    }
	    // an identifier to tell ZoneTask do not create a new invoke closure
	    var OPTIMIZED_ZONE_EVENT_TASK_DATA = {
	        useG: true
	    };
	    var zoneSymbolEventNames = {};
	    var globalSources = {};
	    var EVENT_NAME_SYMBOL_REGX = new RegExp('^' + ZONE_SYMBOL_PREFIX + '(\\w+)(true|false)$');
	    var IMMEDIATE_PROPAGATION_SYMBOL = zoneSymbol$1('propagationStopped');
	    function prepareEventNames(eventName, eventNameToString) {
	        var falseEventName = (eventNameToString ? eventNameToString(eventName) : eventName) + FALSE_STR;
	        var trueEventName = (eventNameToString ? eventNameToString(eventName) : eventName) + TRUE_STR;
	        var symbol = ZONE_SYMBOL_PREFIX + falseEventName;
	        var symbolCapture = ZONE_SYMBOL_PREFIX + trueEventName;
	        zoneSymbolEventNames[eventName] = {};
	        zoneSymbolEventNames[eventName][FALSE_STR] = symbol;
	        zoneSymbolEventNames[eventName][TRUE_STR] = symbolCapture;
	    }
	    function patchEventTarget(_global, api, apis, patchOptions) {
	        var ADD_EVENT_LISTENER = (patchOptions && patchOptions.add) || ADD_EVENT_LISTENER_STR;
	        var REMOVE_EVENT_LISTENER = (patchOptions && patchOptions.rm) || REMOVE_EVENT_LISTENER_STR;
	        var LISTENERS_EVENT_LISTENER = (patchOptions && patchOptions.listeners) || 'eventListeners';
	        var REMOVE_ALL_LISTENERS_EVENT_LISTENER = (patchOptions && patchOptions.rmAll) || 'removeAllListeners';
	        var zoneSymbolAddEventListener = zoneSymbol$1(ADD_EVENT_LISTENER);
	        var ADD_EVENT_LISTENER_SOURCE = '.' + ADD_EVENT_LISTENER + ':';
	        var PREPEND_EVENT_LISTENER = 'prependListener';
	        var PREPEND_EVENT_LISTENER_SOURCE = '.' + PREPEND_EVENT_LISTENER + ':';
	        var invokeTask = function (task, target, event) {
	            // for better performance, check isRemoved which is set
	            // by removeEventListener
	            if (task.isRemoved) {
	                return;
	            }
	            var delegate = task.callback;
	            if (typeof delegate === 'object' && delegate.handleEvent) {
	                // create the bind version of handleEvent when invoke
	                task.callback = function (event) { return delegate.handleEvent(event); };
	                task.originalDelegate = delegate;
	            }
	            // invoke static task.invoke
	            // need to try/catch error here, otherwise, the error in one event listener
	            // will break the executions of the other event listeners. Also error will
	            // not remove the event listener when `once` options is true.
	            var error;
	            try {
	                task.invoke(task, target, [event]);
	            }
	            catch (err) {
	                error = err;
	            }
	            var options = task.options;
	            if (options && typeof options === 'object' && options.once) {
	                // if options.once is true, after invoke once remove listener here
	                // only browser need to do this, nodejs eventEmitter will cal removeListener
	                // inside EventEmitter.once
	                var delegate_1 = task.originalDelegate ? task.originalDelegate : task.callback;
	                target[REMOVE_EVENT_LISTENER].call(target, event.type, delegate_1, options);
	            }
	            return error;
	        };
	        function globalCallback(context, event, isCapture) {
	            // https://github.com/angular/zone.js/issues/911, in IE, sometimes
	            // event will be undefined, so we need to use window.event
	            event = event || _global.event;
	            if (!event) {
	                return;
	            }
	            // event.target is needed for Samsung TV and SourceBuffer
	            // || global is needed https://github.com/angular/zone.js/issues/190
	            var target = context || event.target || _global;
	            var tasks = target[zoneSymbolEventNames[event.type][isCapture ? TRUE_STR : FALSE_STR]];
	            if (tasks) {
	                var errors = [];
	                // invoke all tasks which attached to current target with given event.type and capture = false
	                // for performance concern, if task.length === 1, just invoke
	                if (tasks.length === 1) {
	                    var err = invokeTask(tasks[0], target, event);
	                    err && errors.push(err);
	                }
	                else {
	                    // https://github.com/angular/zone.js/issues/836
	                    // copy the tasks array before invoke, to avoid
	                    // the callback will remove itself or other listener
	                    var copyTasks = tasks.slice();
	                    for (var i = 0; i < copyTasks.length; i++) {
	                        if (event && event[IMMEDIATE_PROPAGATION_SYMBOL] === true) {
	                            break;
	                        }
	                        var err = invokeTask(copyTasks[i], target, event);
	                        err && errors.push(err);
	                    }
	                }
	                // Since there is only one error, we don't need to schedule microTask
	                // to throw the error.
	                if (errors.length === 1) {
	                    throw errors[0];
	                }
	                else {
	                    var _loop_4 = function (i) {
	                        var err = errors[i];
	                        api.nativeScheduleMicroTask(function () {
	                            throw err;
	                        });
	                    };
	                    for (var i = 0; i < errors.length; i++) {
	                        _loop_4(i);
	                    }
	                }
	            }
	        }
	        // global shared zoneAwareCallback to handle all event callback with capture = false
	        var globalZoneAwareCallback = function (event) {
	            return globalCallback(this, event, false);
	        };
	        // global shared zoneAwareCallback to handle all event callback with capture = true
	        var globalZoneAwareCaptureCallback = function (event) {
	            return globalCallback(this, event, true);
	        };
	        function patchEventTargetMethods(obj, patchOptions) {
	            if (!obj) {
	                return false;
	            }
	            var useGlobalCallback = true;
	            if (patchOptions && patchOptions.useG !== undefined) {
	                useGlobalCallback = patchOptions.useG;
	            }
	            var validateHandler = patchOptions && patchOptions.vh;
	            var checkDuplicate = true;
	            if (patchOptions && patchOptions.chkDup !== undefined) {
	                checkDuplicate = patchOptions.chkDup;
	            }
	            var returnTarget = false;
	            if (patchOptions && patchOptions.rt !== undefined) {
	                returnTarget = patchOptions.rt;
	            }
	            var proto = obj;
	            while (proto && !proto.hasOwnProperty(ADD_EVENT_LISTENER)) {
	                proto = ObjectGetPrototypeOf(proto);
	            }
	            if (!proto && obj[ADD_EVENT_LISTENER]) {
	                // somehow we did not find it, but we can see it. This happens on IE for Window properties.
	                proto = obj;
	            }
	            if (!proto) {
	                return false;
	            }
	            if (proto[zoneSymbolAddEventListener]) {
	                return false;
	            }
	            var eventNameToString = patchOptions && patchOptions.eventNameToString;
	            // a shared global taskData to pass data for scheduleEventTask
	            // so we do not need to create a new object just for pass some data
	            var taskData = {};
	            var nativeAddEventListener = proto[zoneSymbolAddEventListener] = proto[ADD_EVENT_LISTENER];
	            var nativeRemoveEventListener = proto[zoneSymbol$1(REMOVE_EVENT_LISTENER)] =
	                proto[REMOVE_EVENT_LISTENER];
	            var nativeListeners = proto[zoneSymbol$1(LISTENERS_EVENT_LISTENER)] =
	                proto[LISTENERS_EVENT_LISTENER];
	            var nativeRemoveAllListeners = proto[zoneSymbol$1(REMOVE_ALL_LISTENERS_EVENT_LISTENER)] =
	                proto[REMOVE_ALL_LISTENERS_EVENT_LISTENER];
	            var nativePrependEventListener;
	            if (patchOptions && patchOptions.prepend) {
	                nativePrependEventListener = proto[zoneSymbol$1(patchOptions.prepend)] =
	                    proto[patchOptions.prepend];
	            }
	            /**
	             * This util function will build an option object with passive option
	             * to handle all possible input from the user.
	             */
	            function buildEventListenerOptions(options, passive) {
	                if (!passiveSupported && typeof options === 'object' && options) {
	                    // doesn't support passive but user want to pass an object as options.
	                    // this will not work on some old browser, so we just pass a boolean
	                    // as useCapture parameter
	                    return !!options.capture;
	                }
	                if (!passiveSupported || !passive) {
	                    return options;
	                }
	                if (typeof options === 'boolean') {
	                    return { capture: options, passive: true };
	                }
	                if (!options) {
	                    return { passive: true };
	                }
	                if (typeof options === 'object' && options.passive !== false) {
	                    return Object.assign(Object.assign({}, options), { passive: true });
	                }
	                return options;
	            }
	            var customScheduleGlobal = function (task) {
	                // if there is already a task for the eventName + capture,
	                // just return, because we use the shared globalZoneAwareCallback here.
	                if (taskData.isExisting) {
	                    return;
	                }
	                return nativeAddEventListener.call(taskData.target, taskData.eventName, taskData.capture ? globalZoneAwareCaptureCallback : globalZoneAwareCallback, taskData.options);
	            };
	            var customCancelGlobal = function (task) {
	                // if task is not marked as isRemoved, this call is directly
	                // from Zone.prototype.cancelTask, we should remove the task
	                // from tasksList of target first
	                if (!task.isRemoved) {
	                    var symbolEventNames = zoneSymbolEventNames[task.eventName];
	                    var symbolEventName = void 0;
	                    if (symbolEventNames) {
	                        symbolEventName = symbolEventNames[task.capture ? TRUE_STR : FALSE_STR];
	                    }
	                    var existingTasks = symbolEventName && task.target[symbolEventName];
	                    if (existingTasks) {
	                        for (var i = 0; i < existingTasks.length; i++) {
	                            var existingTask = existingTasks[i];
	                            if (existingTask === task) {
	                                existingTasks.splice(i, 1);
	                                // set isRemoved to data for faster invokeTask check
	                                task.isRemoved = true;
	                                if (existingTasks.length === 0) {
	                                    // all tasks for the eventName + capture have gone,
	                                    // remove globalZoneAwareCallback and remove the task cache from target
	                                    task.allRemoved = true;
	                                    task.target[symbolEventName] = null;
	                                }
	                                break;
	                            }
	                        }
	                    }
	                }
	                // if all tasks for the eventName + capture have gone,
	                // we will really remove the global event callback,
	                // if not, return
	                if (!task.allRemoved) {
	                    return;
	                }
	                return nativeRemoveEventListener.call(task.target, task.eventName, task.capture ? globalZoneAwareCaptureCallback : globalZoneAwareCallback, task.options);
	            };
	            var customScheduleNonGlobal = function (task) {
	                return nativeAddEventListener.call(taskData.target, taskData.eventName, task.invoke, taskData.options);
	            };
	            var customSchedulePrepend = function (task) {
	                return nativePrependEventListener.call(taskData.target, taskData.eventName, task.invoke, taskData.options);
	            };
	            var customCancelNonGlobal = function (task) {
	                return nativeRemoveEventListener.call(task.target, task.eventName, task.invoke, task.options);
	            };
	            var customSchedule = useGlobalCallback ? customScheduleGlobal : customScheduleNonGlobal;
	            var customCancel = useGlobalCallback ? customCancelGlobal : customCancelNonGlobal;
	            var compareTaskCallbackVsDelegate = function (task, delegate) {
	                var typeOfDelegate = typeof delegate;
	                return (typeOfDelegate === 'function' && task.callback === delegate) ||
	                    (typeOfDelegate === 'object' && task.originalDelegate === delegate);
	            };
	            var compare = (patchOptions && patchOptions.diff) ? patchOptions.diff : compareTaskCallbackVsDelegate;
	            var unpatchedEvents = Zone[zoneSymbol$1('UNPATCHED_EVENTS')];
	            var passiveEvents = _global[zoneSymbol$1('PASSIVE_EVENTS')];
	            var makeAddListener = function (nativeListener, addSource, customScheduleFn, customCancelFn, returnTarget, prepend) {
	                if (returnTarget === void 0) { returnTarget = false; }
	                if (prepend === void 0) { prepend = false; }
	                return function () {
	                    var target = this || _global;
	                    var eventName = arguments[0];
	                    if (patchOptions && patchOptions.transferEventName) {
	                        eventName = patchOptions.transferEventName(eventName);
	                    }
	                    var delegate = arguments[1];
	                    if (!delegate) {
	                        return nativeListener.apply(this, arguments);
	                    }
	                    if (isNode && eventName === 'uncaughtException') {
	                        // don't patch uncaughtException of nodejs to prevent endless loop
	                        return nativeListener.apply(this, arguments);
	                    }
	                    // don't create the bind delegate function for handleEvent
	                    // case here to improve addEventListener performance
	                    // we will create the bind delegate when invoke
	                    var isHandleEvent = false;
	                    if (typeof delegate !== 'function') {
	                        if (!delegate.handleEvent) {
	                            return nativeListener.apply(this, arguments);
	                        }
	                        isHandleEvent = true;
	                    }
	                    if (validateHandler && !validateHandler(nativeListener, delegate, target, arguments)) {
	                        return;
	                    }
	                    var passive = passiveSupported && !!passiveEvents && passiveEvents.indexOf(eventName) !== -1;
	                    var options = buildEventListenerOptions(arguments[2], passive);
	                    if (unpatchedEvents) {
	                        // check unpatched list
	                        for (var i = 0; i < unpatchedEvents.length; i++) {
	                            if (eventName === unpatchedEvents[i]) {
	                                if (passive) {
	                                    return nativeListener.call(target, eventName, delegate, options);
	                                }
	                                else {
	                                    return nativeListener.apply(this, arguments);
	                                }
	                            }
	                        }
	                    }
	                    var capture = !options ? false : typeof options === 'boolean' ? true : options.capture;
	                    var once = options && typeof options === 'object' ? options.once : false;
	                    var zone = Zone.current;
	                    var symbolEventNames = zoneSymbolEventNames[eventName];
	                    if (!symbolEventNames) {
	                        prepareEventNames(eventName, eventNameToString);
	                        symbolEventNames = zoneSymbolEventNames[eventName];
	                    }
	                    var symbolEventName = symbolEventNames[capture ? TRUE_STR : FALSE_STR];
	                    var existingTasks = target[symbolEventName];
	                    var isExisting = false;
	                    if (existingTasks) {
	                        // already have task registered
	                        isExisting = true;
	                        if (checkDuplicate) {
	                            for (var i = 0; i < existingTasks.length; i++) {
	                                if (compare(existingTasks[i], delegate)) {
	                                    // same callback, same capture, same event name, just return
	                                    return;
	                                }
	                            }
	                        }
	                    }
	                    else {
	                        existingTasks = target[symbolEventName] = [];
	                    }
	                    var source;
	                    var constructorName = target.constructor['name'];
	                    var targetSource = globalSources[constructorName];
	                    if (targetSource) {
	                        source = targetSource[eventName];
	                    }
	                    if (!source) {
	                        source = constructorName + addSource +
	                            (eventNameToString ? eventNameToString(eventName) : eventName);
	                    }
	                    // do not create a new object as task.data to pass those things
	                    // just use the global shared one
	                    taskData.options = options;
	                    if (once) {
	                        // if addEventListener with once options, we don't pass it to
	                        // native addEventListener, instead we keep the once setting
	                        // and handle ourselves.
	                        taskData.options.once = false;
	                    }
	                    taskData.target = target;
	                    taskData.capture = capture;
	                    taskData.eventName = eventName;
	                    taskData.isExisting = isExisting;
	                    var data = useGlobalCallback ? OPTIMIZED_ZONE_EVENT_TASK_DATA : undefined;
	                    // keep taskData into data to allow onScheduleEventTask to access the task information
	                    if (data) {
	                        data.taskData = taskData;
	                    }
	                    var task = zone.scheduleEventTask(source, delegate, data, customScheduleFn, customCancelFn);
	                    // should clear taskData.target to avoid memory leak
	                    // issue, https://github.com/angular/angular/issues/20442
	                    taskData.target = null;
	                    // need to clear up taskData because it is a global object
	                    if (data) {
	                        data.taskData = null;
	                    }
	                    // have to save those information to task in case
	                    // application may call task.zone.cancelTask() directly
	                    if (once) {
	                        options.once = true;
	                    }
	                    if (!(!passiveSupported && typeof task.options === 'boolean')) {
	                        // if not support passive, and we pass an option object
	                        // to addEventListener, we should save the options to task
	                        task.options = options;
	                    }
	                    task.target = target;
	                    task.capture = capture;
	                    task.eventName = eventName;
	                    if (isHandleEvent) {
	                        // save original delegate for compare to check duplicate
	                        task.originalDelegate = delegate;
	                    }
	                    if (!prepend) {
	                        existingTasks.push(task);
	                    }
	                    else {
	                        existingTasks.unshift(task);
	                    }
	                    if (returnTarget) {
	                        return target;
	                    }
	                };
	            };
	            proto[ADD_EVENT_LISTENER] = makeAddListener(nativeAddEventListener, ADD_EVENT_LISTENER_SOURCE, customSchedule, customCancel, returnTarget);
	            if (nativePrependEventListener) {
	                proto[PREPEND_EVENT_LISTENER] = makeAddListener(nativePrependEventListener, PREPEND_EVENT_LISTENER_SOURCE, customSchedulePrepend, customCancel, returnTarget, true);
	            }
	            proto[REMOVE_EVENT_LISTENER] = function () {
	                var target = this || _global;
	                var eventName = arguments[0];
	                if (patchOptions && patchOptions.transferEventName) {
	                    eventName = patchOptions.transferEventName(eventName);
	                }
	                var options = arguments[2];
	                var capture = !options ? false : typeof options === 'boolean' ? true : options.capture;
	                var delegate = arguments[1];
	                if (!delegate) {
	                    return nativeRemoveEventListener.apply(this, arguments);
	                }
	                if (validateHandler &&
	                    !validateHandler(nativeRemoveEventListener, delegate, target, arguments)) {
	                    return;
	                }
	                var symbolEventNames = zoneSymbolEventNames[eventName];
	                var symbolEventName;
	                if (symbolEventNames) {
	                    symbolEventName = symbolEventNames[capture ? TRUE_STR : FALSE_STR];
	                }
	                var existingTasks = symbolEventName && target[symbolEventName];
	                if (existingTasks) {
	                    for (var i = 0; i < existingTasks.length; i++) {
	                        var existingTask = existingTasks[i];
	                        if (compare(existingTask, delegate)) {
	                            existingTasks.splice(i, 1);
	                            // set isRemoved to data for faster invokeTask check
	                            existingTask.isRemoved = true;
	                            if (existingTasks.length === 0) {
	                                // all tasks for the eventName + capture have gone,
	                                // remove globalZoneAwareCallback and remove the task cache from target
	                                existingTask.allRemoved = true;
	                                target[symbolEventName] = null;
	                                // in the target, we have an event listener which is added by on_property
	                                // such as target.onclick = function() {}, so we need to clear this internal
	                                // property too if all delegates all removed
	                                if (typeof eventName === 'string') {
	                                    var onPropertySymbol = ZONE_SYMBOL_PREFIX + 'ON_PROPERTY' + eventName;
	                                    target[onPropertySymbol] = null;
	                                }
	                            }
	                            existingTask.zone.cancelTask(existingTask);
	                            if (returnTarget) {
	                                return target;
	                            }
	                            return;
	                        }
	                    }
	                }
	                // issue 930, didn't find the event name or callback
	                // from zone kept existingTasks, the callback maybe
	                // added outside of zone, we need to call native removeEventListener
	                // to try to remove it.
	                return nativeRemoveEventListener.apply(this, arguments);
	            };
	            proto[LISTENERS_EVENT_LISTENER] = function () {
	                var target = this || _global;
	                var eventName = arguments[0];
	                if (patchOptions && patchOptions.transferEventName) {
	                    eventName = patchOptions.transferEventName(eventName);
	                }
	                var listeners = [];
	                var tasks = findEventTasks(target, eventNameToString ? eventNameToString(eventName) : eventName);
	                for (var i = 0; i < tasks.length; i++) {
	                    var task = tasks[i];
	                    var delegate = task.originalDelegate ? task.originalDelegate : task.callback;
	                    listeners.push(delegate);
	                }
	                return listeners;
	            };
	            proto[REMOVE_ALL_LISTENERS_EVENT_LISTENER] = function () {
	                var target = this || _global;
	                var eventName = arguments[0];
	                if (!eventName) {
	                    var keys = Object.keys(target);
	                    for (var i = 0; i < keys.length; i++) {
	                        var prop = keys[i];
	                        var match = EVENT_NAME_SYMBOL_REGX.exec(prop);
	                        var evtName = match && match[1];
	                        // in nodejs EventEmitter, removeListener event is
	                        // used for monitoring the removeListener call,
	                        // so just keep removeListener eventListener until
	                        // all other eventListeners are removed
	                        if (evtName && evtName !== 'removeListener') {
	                            this[REMOVE_ALL_LISTENERS_EVENT_LISTENER].call(this, evtName);
	                        }
	                    }
	                    // remove removeListener listener finally
	                    this[REMOVE_ALL_LISTENERS_EVENT_LISTENER].call(this, 'removeListener');
	                }
	                else {
	                    if (patchOptions && patchOptions.transferEventName) {
	                        eventName = patchOptions.transferEventName(eventName);
	                    }
	                    var symbolEventNames = zoneSymbolEventNames[eventName];
	                    if (symbolEventNames) {
	                        var symbolEventName = symbolEventNames[FALSE_STR];
	                        var symbolCaptureEventName = symbolEventNames[TRUE_STR];
	                        var tasks = target[symbolEventName];
	                        var captureTasks = target[symbolCaptureEventName];
	                        if (tasks) {
	                            var removeTasks = tasks.slice();
	                            for (var i = 0; i < removeTasks.length; i++) {
	                                var task = removeTasks[i];
	                                var delegate = task.originalDelegate ? task.originalDelegate : task.callback;
	                                this[REMOVE_EVENT_LISTENER].call(this, eventName, delegate, task.options);
	                            }
	                        }
	                        if (captureTasks) {
	                            var removeTasks = captureTasks.slice();
	                            for (var i = 0; i < removeTasks.length; i++) {
	                                var task = removeTasks[i];
	                                var delegate = task.originalDelegate ? task.originalDelegate : task.callback;
	                                this[REMOVE_EVENT_LISTENER].call(this, eventName, delegate, task.options);
	                            }
	                        }
	                    }
	                }
	                if (returnTarget) {
	                    return this;
	                }
	            };
	            // for native toString patch
	            attachOriginToPatched(proto[ADD_EVENT_LISTENER], nativeAddEventListener);
	            attachOriginToPatched(proto[REMOVE_EVENT_LISTENER], nativeRemoveEventListener);
	            if (nativeRemoveAllListeners) {
	                attachOriginToPatched(proto[REMOVE_ALL_LISTENERS_EVENT_LISTENER], nativeRemoveAllListeners);
	            }
	            if (nativeListeners) {
	                attachOriginToPatched(proto[LISTENERS_EVENT_LISTENER], nativeListeners);
	            }
	            return true;
	        }
	        var results = [];
	        for (var i = 0; i < apis.length; i++) {
	            results[i] = patchEventTargetMethods(apis[i], patchOptions);
	        }
	        return results;
	    }
	    function findEventTasks(target, eventName) {
	        if (!eventName) {
	            var foundTasks = [];
	            for (var prop in target) {
	                var match = EVENT_NAME_SYMBOL_REGX.exec(prop);
	                var evtName = match && match[1];
	                if (evtName && (!eventName || evtName === eventName)) {
	                    var tasks = target[prop];
	                    if (tasks) {
	                        for (var i = 0; i < tasks.length; i++) {
	                            foundTasks.push(tasks[i]);
	                        }
	                    }
	                }
	            }
	            return foundTasks;
	        }
	        var symbolEventName = zoneSymbolEventNames[eventName];
	        if (!symbolEventName) {
	            prepareEventNames(eventName);
	            symbolEventName = zoneSymbolEventNames[eventName];
	        }
	        var captureFalseTasks = target[symbolEventName[FALSE_STR]];
	        var captureTrueTasks = target[symbolEventName[TRUE_STR]];
	        if (!captureFalseTasks) {
	            return captureTrueTasks ? captureTrueTasks.slice() : [];
	        }
	        else {
	            return captureTrueTasks ? captureFalseTasks.concat(captureTrueTasks) :
	                captureFalseTasks.slice();
	        }
	    }
	    function patchEventPrototype(global, api) {
	        var Event = global['Event'];
	        if (Event && Event.prototype) {
	            api.patchMethod(Event.prototype, 'stopImmediatePropagation', function (delegate) { return function (self, args) {
	                self[IMMEDIATE_PROPAGATION_SYMBOL] = true;
	                // we need to call the native stopImmediatePropagation
	                // in case in some hybrid application, some part of
	                // application will be controlled by zone, some are not
	                delegate && delegate.apply(self, args);
	            }; });
	        }
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function patchCallbacks(api, target, targetName, method, callbacks) {
	        var symbol = Zone.__symbol__(method);
	        if (target[symbol]) {
	            return;
	        }
	        var nativeDelegate = target[symbol] = target[method];
	        target[method] = function (name, opts, options) {
	            if (opts && opts.prototype) {
	                callbacks.forEach(function (callback) {
	                    var source = "".concat(targetName, ".").concat(method, "::") + callback;
	                    var prototype = opts.prototype;
	                    // Note: the `patchCallbacks` is used for patching the `document.registerElement` and
	                    // `customElements.define`. We explicitly wrap the patching code into try-catch since
	                    // callbacks may be already patched by other web components frameworks (e.g. LWC), and they
	                    // make those properties non-writable. This means that patching callback will throw an error
	                    // `cannot assign to read-only property`. See this code as an example:
	                    // https://github.com/salesforce/lwc/blob/master/packages/@lwc/engine-core/src/framework/base-bridge-element.ts#L180-L186
	                    // We don't want to stop the application rendering if we couldn't patch some
	                    // callback, e.g. `attributeChangedCallback`.
	                    try {
	                        if (prototype.hasOwnProperty(callback)) {
	                            var descriptor = api.ObjectGetOwnPropertyDescriptor(prototype, callback);
	                            if (descriptor && descriptor.value) {
	                                descriptor.value = api.wrapWithCurrentZone(descriptor.value, source);
	                                api._redefineProperty(opts.prototype, callback, descriptor);
	                            }
	                            else if (prototype[callback]) {
	                                prototype[callback] = api.wrapWithCurrentZone(prototype[callback], source);
	                            }
	                        }
	                        else if (prototype[callback]) {
	                            prototype[callback] = api.wrapWithCurrentZone(prototype[callback], source);
	                        }
	                    }
	                    catch (_a) {
	                        // Note: we leave the catch block empty since there's no way to handle the error related
	                        // to non-writable property.
	                    }
	                });
	            }
	            return nativeDelegate.call(target, name, opts, options);
	        };
	        api.attachOriginToPatched(target[method], nativeDelegate);
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function filterProperties(target, onProperties, ignoreProperties) {
	        if (!ignoreProperties || ignoreProperties.length === 0) {
	            return onProperties;
	        }
	        var tip = ignoreProperties.filter(function (ip) { return ip.target === target; });
	        if (!tip || tip.length === 0) {
	            return onProperties;
	        }
	        var targetIgnoreProperties = tip[0].ignoreProperties;
	        return onProperties.filter(function (op) { return targetIgnoreProperties.indexOf(op) === -1; });
	    }
	    function patchFilteredProperties(target, onProperties, ignoreProperties, prototype) {
	        // check whether target is available, sometimes target will be undefined
	        // because different browser or some 3rd party plugin.
	        if (!target) {
	            return;
	        }
	        var filteredProperties = filterProperties(target, onProperties, ignoreProperties);
	        patchOnProperties(target, filteredProperties, prototype);
	    }
	    /**
	     * Get all event name properties which the event name startsWith `on`
	     * from the target object itself, inherited properties are not considered.
	     */
	    function getOnEventNames(target) {
	        return Object.getOwnPropertyNames(target)
	            .filter(function (name) { return name.startsWith('on') && name.length > 2; })
	            .map(function (name) { return name.substring(2); });
	    }
	    function propertyDescriptorPatch(api, _global) {
	        if (isNode && !isMix) {
	            return;
	        }
	        if (Zone[api.symbol('patchEvents')]) {
	            // events are already been patched by legacy patch.
	            return;
	        }
	        var ignoreProperties = _global['__Zone_ignore_on_properties'];
	        // for browsers that we can patch the descriptor:  Chrome & Firefox
	        var patchTargets = [];
	        if (isBrowser) {
	            var internalWindow_1 = window;
	            patchTargets = patchTargets.concat([
	                'Document', 'SVGElement', 'Element', 'HTMLElement', 'HTMLBodyElement', 'HTMLMediaElement',
	                'HTMLFrameSetElement', 'HTMLFrameElement', 'HTMLIFrameElement', 'HTMLMarqueeElement', 'Worker'
	            ]);
	            var ignoreErrorProperties = isIE() ? [{ target: internalWindow_1, ignoreProperties: ['error'] }] : [];
	            // in IE/Edge, onProp not exist in window object, but in WindowPrototype
	            // so we need to pass WindowPrototype to check onProp exist or not
	            patchFilteredProperties(internalWindow_1, getOnEventNames(internalWindow_1), ignoreProperties ? ignoreProperties.concat(ignoreErrorProperties) : ignoreProperties, ObjectGetPrototypeOf(internalWindow_1));
	        }
	        patchTargets = patchTargets.concat([
	            'XMLHttpRequest', 'XMLHttpRequestEventTarget', 'IDBIndex', 'IDBRequest', 'IDBOpenDBRequest',
	            'IDBDatabase', 'IDBTransaction', 'IDBCursor', 'WebSocket'
	        ]);
	        for (var i = 0; i < patchTargets.length; i++) {
	            var target = _global[patchTargets[i]];
	            target && target.prototype &&
	                patchFilteredProperties(target.prototype, getOnEventNames(target.prototype), ignoreProperties);
	        }
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    Zone.__load_patch('util', function (global, Zone, api) {
	        // Collect native event names by looking at properties
	        // on the global namespace, e.g. 'onclick'.
	        var eventNames = getOnEventNames(global);
	        api.patchOnProperties = patchOnProperties;
	        api.patchMethod = patchMethod;
	        api.bindArguments = bindArguments;
	        api.patchMacroTask = patchMacroTask;
	        // In earlier version of zone.js (<0.9.0), we use env name `__zone_symbol__BLACK_LISTED_EVENTS` to
	        // define which events will not be patched by `Zone.js`.
	        // In newer version (>=0.9.0), we change the env name to `__zone_symbol__UNPATCHED_EVENTS` to keep
	        // the name consistent with angular repo.
	        // The  `__zone_symbol__BLACK_LISTED_EVENTS` is deprecated, but it is still be supported for
	        // backwards compatibility.
	        var SYMBOL_BLACK_LISTED_EVENTS = Zone.__symbol__('BLACK_LISTED_EVENTS');
	        var SYMBOL_UNPATCHED_EVENTS = Zone.__symbol__('UNPATCHED_EVENTS');
	        if (global[SYMBOL_UNPATCHED_EVENTS]) {
	            global[SYMBOL_BLACK_LISTED_EVENTS] = global[SYMBOL_UNPATCHED_EVENTS];
	        }
	        if (global[SYMBOL_BLACK_LISTED_EVENTS]) {
	            Zone[SYMBOL_BLACK_LISTED_EVENTS] = Zone[SYMBOL_UNPATCHED_EVENTS] =
	                global[SYMBOL_BLACK_LISTED_EVENTS];
	        }
	        api.patchEventPrototype = patchEventPrototype;
	        api.patchEventTarget = patchEventTarget;
	        api.isIEOrEdge = isIEOrEdge;
	        api.ObjectDefineProperty = ObjectDefineProperty;
	        api.ObjectGetOwnPropertyDescriptor = ObjectGetOwnPropertyDescriptor;
	        api.ObjectCreate = ObjectCreate;
	        api.ArraySlice = ArraySlice;
	        api.patchClass = patchClass;
	        api.wrapWithCurrentZone = wrapWithCurrentZone;
	        api.filterProperties = filterProperties;
	        api.attachOriginToPatched = attachOriginToPatched;
	        api._redefineProperty = Object.defineProperty;
	        api.patchCallbacks = patchCallbacks;
	        api.getGlobalObjects = function () { return ({
	            globalSources: globalSources,
	            zoneSymbolEventNames: zoneSymbolEventNames,
	            eventNames: eventNames,
	            isBrowser: isBrowser,
	            isMix: isMix,
	            isNode: isNode,
	            TRUE_STR: TRUE_STR,
	            FALSE_STR: FALSE_STR,
	            ZONE_SYMBOL_PREFIX: ZONE_SYMBOL_PREFIX,
	            ADD_EVENT_LISTENER_STR: ADD_EVENT_LISTENER_STR,
	            REMOVE_EVENT_LISTENER_STR: REMOVE_EVENT_LISTENER_STR
	        }); };
	    });
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    /*
	     * This is necessary for Chrome and Chrome mobile, to enable
	     * things like redefining `createdCallback` on an element.
	     */
	    var zoneSymbol;
	    var _defineProperty;
	    var _getOwnPropertyDescriptor;
	    var _create;
	    var unconfigurablesKey;
	    function propertyPatch() {
	        zoneSymbol = Zone.__symbol__;
	        _defineProperty = Object[zoneSymbol('defineProperty')] = Object.defineProperty;
	        _getOwnPropertyDescriptor = Object[zoneSymbol('getOwnPropertyDescriptor')] =
	            Object.getOwnPropertyDescriptor;
	        _create = Object.create;
	        unconfigurablesKey = zoneSymbol('unconfigurables');
	        Object.defineProperty = function (obj, prop, desc) {
	            if (isUnconfigurable(obj, prop)) {
	                throw new TypeError('Cannot assign to read only property \'' + prop + '\' of ' + obj);
	            }
	            var originalConfigurableFlag = desc.configurable;
	            if (prop !== 'prototype') {
	                desc = rewriteDescriptor(obj, prop, desc);
	            }
	            return _tryDefineProperty(obj, prop, desc, originalConfigurableFlag);
	        };
	        Object.defineProperties = function (obj, props) {
	            Object.keys(props).forEach(function (prop) {
	                Object.defineProperty(obj, prop, props[prop]);
	            });
	            for (var _i = 0, _b = Object.getOwnPropertySymbols(props); _i < _b.length; _i++) {
	                var sym = _b[_i];
	                var desc = Object.getOwnPropertyDescriptor(props, sym);
	                // Since `Object.getOwnPropertySymbols` returns *all* symbols,
	                // including non-enumerable ones, retrieve property descriptor and check
	                // enumerability there. Proceed with the rewrite only when a property is
	                // enumerable to make the logic consistent with the way regular
	                // properties are retrieved (via `Object.keys`, which respects
	                // `enumerable: false` flag). More information:
	                // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Enumerability_and_ownership_of_properties#retrieval
	                if (desc === null || desc === void 0 ? void 0 : desc.enumerable) {
	                    Object.defineProperty(obj, sym, props[sym]);
	                }
	            }
	            return obj;
	        };
	        Object.create = function (proto, propertiesObject) {
	            if (typeof propertiesObject === 'object' && !Object.isFrozen(propertiesObject)) {
	                Object.keys(propertiesObject).forEach(function (prop) {
	                    propertiesObject[prop] = rewriteDescriptor(proto, prop, propertiesObject[prop]);
	                });
	            }
	            return _create(proto, propertiesObject);
	        };
	        Object.getOwnPropertyDescriptor = function (obj, prop) {
	            var desc = _getOwnPropertyDescriptor(obj, prop);
	            if (desc && isUnconfigurable(obj, prop)) {
	                desc.configurable = false;
	            }
	            return desc;
	        };
	    }
	    function _redefineProperty(obj, prop, desc) {
	        var originalConfigurableFlag = desc.configurable;
	        desc = rewriteDescriptor(obj, prop, desc);
	        return _tryDefineProperty(obj, prop, desc, originalConfigurableFlag);
	    }
	    function isUnconfigurable(obj, prop) {
	        return obj && obj[unconfigurablesKey] && obj[unconfigurablesKey][prop];
	    }
	    function rewriteDescriptor(obj, prop, desc) {
	        // issue-927, if the desc is frozen, don't try to change the desc
	        if (!Object.isFrozen(desc)) {
	            desc.configurable = true;
	        }
	        if (!desc.configurable) {
	            // issue-927, if the obj is frozen, don't try to set the desc to obj
	            if (!obj[unconfigurablesKey] && !Object.isFrozen(obj)) {
	                _defineProperty(obj, unconfigurablesKey, { writable: true, value: {} });
	            }
	            if (obj[unconfigurablesKey]) {
	                obj[unconfigurablesKey][prop] = true;
	            }
	        }
	        return desc;
	    }
	    function _tryDefineProperty(obj, prop, desc, originalConfigurableFlag) {
	        try {
	            return _defineProperty(obj, prop, desc);
	        }
	        catch (error) {
	            if (desc.configurable) {
	                // In case of errors, when the configurable flag was likely set by rewriteDescriptor(),
	                // let's retry with the original flag value
	                if (typeof originalConfigurableFlag == 'undefined') {
	                    delete desc.configurable;
	                }
	                else {
	                    desc.configurable = originalConfigurableFlag;
	                }
	                try {
	                    return _defineProperty(obj, prop, desc);
	                }
	                catch (error) {
	                    var swallowError = false;
	                    if (prop === 'createdCallback' || prop === 'attachedCallback' ||
	                        prop === 'detachedCallback' || prop === 'attributeChangedCallback') {
	                        // We only swallow the error in registerElement patch
	                        // this is the work around since some applications
	                        // fail if we throw the error
	                        swallowError = true;
	                    }
	                    if (!swallowError) {
	                        throw error;
	                    }
	                    // TODO: @JiaLiPassion, Some application such as `registerElement` patch
	                    // still need to swallow the error, in the future after these applications
	                    // are updated, the following logic can be removed.
	                    var descJson = null;
	                    try {
	                        descJson = JSON.stringify(desc);
	                    }
	                    catch (error) {
	                        descJson = desc.toString();
	                    }
	                    console.log("Attempting to configure '".concat(prop, "' with descriptor '").concat(descJson, "' on object '").concat(obj, "' and got error, giving up: ").concat(error));
	                }
	            }
	            else {
	                throw error;
	            }
	        }
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function eventTargetLegacyPatch(_global, api) {
	        var _b = api.getGlobalObjects(), eventNames = _b.eventNames, globalSources = _b.globalSources, zoneSymbolEventNames = _b.zoneSymbolEventNames, TRUE_STR = _b.TRUE_STR, FALSE_STR = _b.FALSE_STR, ZONE_SYMBOL_PREFIX = _b.ZONE_SYMBOL_PREFIX;
	        var WTF_ISSUE_555 = 'Anchor,Area,Audio,BR,Base,BaseFont,Body,Button,Canvas,Content,DList,Directory,Div,Embed,FieldSet,Font,Form,Frame,FrameSet,HR,Head,Heading,Html,IFrame,Image,Input,Keygen,LI,Label,Legend,Link,Map,Marquee,Media,Menu,Meta,Meter,Mod,OList,Object,OptGroup,Option,Output,Paragraph,Pre,Progress,Quote,Script,Select,Source,Span,Style,TableCaption,TableCell,TableCol,Table,TableRow,TableSection,TextArea,Title,Track,UList,Unknown,Video';
	        var NO_EVENT_TARGET = 'ApplicationCache,EventSource,FileReader,InputMethodContext,MediaController,MessagePort,Node,Performance,SVGElementInstance,SharedWorker,TextTrack,TextTrackCue,TextTrackList,WebKitNamedFlow,Window,Worker,WorkerGlobalScope,XMLHttpRequest,XMLHttpRequestEventTarget,XMLHttpRequestUpload,IDBRequest,IDBOpenDBRequest,IDBDatabase,IDBTransaction,IDBCursor,DBIndex,WebSocket'
	            .split(',');
	        var EVENT_TARGET = 'EventTarget';
	        var apis = [];
	        var isWtf = _global['wtf'];
	        var WTF_ISSUE_555_ARRAY = WTF_ISSUE_555.split(',');
	        if (isWtf) {
	            // Workaround for: https://github.com/google/tracing-framework/issues/555
	            apis = WTF_ISSUE_555_ARRAY.map(function (v) { return 'HTML' + v + 'Element'; }).concat(NO_EVENT_TARGET);
	        }
	        else if (_global[EVENT_TARGET]) {
	            apis.push(EVENT_TARGET);
	        }
	        else {
	            // Note: EventTarget is not available in all browsers,
	            // if it's not available, we instead patch the APIs in the IDL that inherit from EventTarget
	            apis = NO_EVENT_TARGET;
	        }
	        var isDisableIECheck = _global['__Zone_disable_IE_check'] || false;
	        var isEnableCrossContextCheck = _global['__Zone_enable_cross_context_check'] || false;
	        var ieOrEdge = api.isIEOrEdge();
	        var ADD_EVENT_LISTENER_SOURCE = '.addEventListener:';
	        var FUNCTION_WRAPPER = '[object FunctionWrapper]';
	        var BROWSER_TOOLS = 'function __BROWSERTOOLS_CONSOLE_SAFEFUNC() { [native code] }';
	        var pointerEventsMap = {
	            'MSPointerCancel': 'pointercancel',
	            'MSPointerDown': 'pointerdown',
	            'MSPointerEnter': 'pointerenter',
	            'MSPointerHover': 'pointerhover',
	            'MSPointerLeave': 'pointerleave',
	            'MSPointerMove': 'pointermove',
	            'MSPointerOut': 'pointerout',
	            'MSPointerOver': 'pointerover',
	            'MSPointerUp': 'pointerup'
	        };
	        //  predefine all __zone_symbol__ + eventName + true/false string
	        for (var i = 0; i < eventNames.length; i++) {
	            var eventName = eventNames[i];
	            var falseEventName = eventName + FALSE_STR;
	            var trueEventName = eventName + TRUE_STR;
	            var symbol = ZONE_SYMBOL_PREFIX + falseEventName;
	            var symbolCapture = ZONE_SYMBOL_PREFIX + trueEventName;
	            zoneSymbolEventNames[eventName] = {};
	            zoneSymbolEventNames[eventName][FALSE_STR] = symbol;
	            zoneSymbolEventNames[eventName][TRUE_STR] = symbolCapture;
	        }
	        //  predefine all task.source string
	        for (var i = 0; i < WTF_ISSUE_555_ARRAY.length; i++) {
	            var target = WTF_ISSUE_555_ARRAY[i];
	            var targets = globalSources[target] = {};
	            for (var j = 0; j < eventNames.length; j++) {
	                var eventName = eventNames[j];
	                targets[eventName] = target + ADD_EVENT_LISTENER_SOURCE + eventName;
	            }
	        }
	        var checkIEAndCrossContext = function (nativeDelegate, delegate, target, args) {
	            if (!isDisableIECheck && ieOrEdge) {
	                if (isEnableCrossContextCheck) {
	                    try {
	                        var testString = delegate.toString();
	                        if ((testString === FUNCTION_WRAPPER || testString == BROWSER_TOOLS)) {
	                            nativeDelegate.apply(target, args);
	                            return false;
	                        }
	                    }
	                    catch (error) {
	                        nativeDelegate.apply(target, args);
	                        return false;
	                    }
	                }
	                else {
	                    var testString = delegate.toString();
	                    if ((testString === FUNCTION_WRAPPER || testString == BROWSER_TOOLS)) {
	                        nativeDelegate.apply(target, args);
	                        return false;
	                    }
	                }
	            }
	            else if (isEnableCrossContextCheck) {
	                try {
	                    delegate.toString();
	                }
	                catch (error) {
	                    nativeDelegate.apply(target, args);
	                    return false;
	                }
	            }
	            return true;
	        };
	        var apiTypes = [];
	        for (var i = 0; i < apis.length; i++) {
	            var type = _global[apis[i]];
	            apiTypes.push(type && type.prototype);
	        }
	        // vh is validateHandler to check event handler
	        // is valid or not(for security check)
	        api.patchEventTarget(_global, api, apiTypes, {
	            vh: checkIEAndCrossContext,
	            transferEventName: function (eventName) {
	                var pointerEventName = pointerEventsMap[eventName];
	                return pointerEventName || eventName;
	            }
	        });
	        Zone[api.symbol('patchEventTarget')] = !!_global[EVENT_TARGET];
	        return true;
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    // we have to patch the instance since the proto is non-configurable
	    function apply(api, _global) {
	        var _b = api.getGlobalObjects(), ADD_EVENT_LISTENER_STR = _b.ADD_EVENT_LISTENER_STR, REMOVE_EVENT_LISTENER_STR = _b.REMOVE_EVENT_LISTENER_STR;
	        var WS = _global.WebSocket;
	        // On Safari window.EventTarget doesn't exist so need to patch WS add/removeEventListener
	        // On older Chrome, no need since EventTarget was already patched
	        if (!_global.EventTarget) {
	            api.patchEventTarget(_global, api, [WS.prototype]);
	        }
	        _global.WebSocket = function (x, y) {
	            var socket = arguments.length > 1 ? new WS(x, y) : new WS(x);
	            var proxySocket;
	            var proxySocketProto;
	            // Safari 7.0 has non-configurable own 'onmessage' and friends properties on the socket instance
	            var onmessageDesc = api.ObjectGetOwnPropertyDescriptor(socket, 'onmessage');
	            if (onmessageDesc && onmessageDesc.configurable === false) {
	                proxySocket = api.ObjectCreate(socket);
	                // socket have own property descriptor 'onopen', 'onmessage', 'onclose', 'onerror'
	                // but proxySocket not, so we will keep socket as prototype and pass it to
	                // patchOnProperties method
	                proxySocketProto = socket;
	                [ADD_EVENT_LISTENER_STR, REMOVE_EVENT_LISTENER_STR, 'send', 'close'].forEach(function (propName) {
	                    proxySocket[propName] = function () {
	                        var args = api.ArraySlice.call(arguments);
	                        if (propName === ADD_EVENT_LISTENER_STR || propName === REMOVE_EVENT_LISTENER_STR) {
	                            var eventName = args.length > 0 ? args[0] : undefined;
	                            if (eventName) {
	                                var propertySymbol = Zone.__symbol__('ON_PROPERTY' + eventName);
	                                socket[propertySymbol] = proxySocket[propertySymbol];
	                            }
	                        }
	                        return socket[propName].apply(socket, args);
	                    };
	                });
	            }
	            else {
	                // we can patch the real socket
	                proxySocket = socket;
	            }
	            api.patchOnProperties(proxySocket, ['close', 'error', 'message', 'open'], proxySocketProto);
	            return proxySocket;
	        };
	        var globalWebSocket = _global['WebSocket'];
	        for (var prop in WS) {
	            globalWebSocket[prop] = WS[prop];
	        }
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function propertyDescriptorLegacyPatch(api, _global) {
	        var _b = api.getGlobalObjects(), isNode = _b.isNode, isMix = _b.isMix;
	        if (isNode && !isMix) {
	            return;
	        }
	        if (!canPatchViaPropertyDescriptor(api, _global)) {
	            var supportsWebSocket = typeof WebSocket !== 'undefined';
	            // Safari, Android browsers (Jelly Bean)
	            patchViaCapturingAllTheEvents(api);
	            api.patchClass('XMLHttpRequest');
	            if (supportsWebSocket) {
	                apply(api, _global);
	            }
	            Zone[api.symbol('patchEvents')] = true;
	        }
	    }
	    function canPatchViaPropertyDescriptor(api, _global) {
	        var _b = api.getGlobalObjects(), isBrowser = _b.isBrowser, isMix = _b.isMix;
	        if ((isBrowser || isMix) &&
	            !api.ObjectGetOwnPropertyDescriptor(HTMLElement.prototype, 'onclick') &&
	            typeof Element !== 'undefined') {
	            // WebKit https://bugs.webkit.org/show_bug.cgi?id=134364
	            // IDL interface attributes are not configurable
	            var desc = api.ObjectGetOwnPropertyDescriptor(Element.prototype, 'onclick');
	            if (desc && !desc.configurable)
	                return false;
	            // try to use onclick to detect whether we can patch via propertyDescriptor
	            // because XMLHttpRequest is not available in service worker
	            if (desc) {
	                api.ObjectDefineProperty(Element.prototype, 'onclick', {
	                    enumerable: true,
	                    configurable: true,
	                    get: function () {
	                        return true;
	                    }
	                });
	                var div = document.createElement('div');
	                var result = !!div.onclick;
	                api.ObjectDefineProperty(Element.prototype, 'onclick', desc);
	                return result;
	            }
	        }
	        var XMLHttpRequest = _global['XMLHttpRequest'];
	        if (!XMLHttpRequest) {
	            // XMLHttpRequest is not available in service worker
	            return false;
	        }
	        var ON_READY_STATE_CHANGE = 'onreadystatechange';
	        var XMLHttpRequestPrototype = XMLHttpRequest.prototype;
	        var xhrDesc = api.ObjectGetOwnPropertyDescriptor(XMLHttpRequestPrototype, ON_READY_STATE_CHANGE);
	        // add enumerable and configurable here because in opera
	        // by default XMLHttpRequest.prototype.onreadystatechange is undefined
	        // without adding enumerable and configurable will cause onreadystatechange
	        // non-configurable
	        // and if XMLHttpRequest.prototype.onreadystatechange is undefined,
	        // we should set a real desc instead a fake one
	        if (xhrDesc) {
	            api.ObjectDefineProperty(XMLHttpRequestPrototype, ON_READY_STATE_CHANGE, {
	                enumerable: true,
	                configurable: true,
	                get: function () {
	                    return true;
	                }
	            });
	            var req = new XMLHttpRequest();
	            var result = !!req.onreadystatechange;
	            // restore original desc
	            api.ObjectDefineProperty(XMLHttpRequestPrototype, ON_READY_STATE_CHANGE, xhrDesc || {});
	            return result;
	        }
	        else {
	            var SYMBOL_FAKE_ONREADYSTATECHANGE_1 = api.symbol('fake');
	            api.ObjectDefineProperty(XMLHttpRequestPrototype, ON_READY_STATE_CHANGE, {
	                enumerable: true,
	                configurable: true,
	                get: function () {
	                    return this[SYMBOL_FAKE_ONREADYSTATECHANGE_1];
	                },
	                set: function (value) {
	                    this[SYMBOL_FAKE_ONREADYSTATECHANGE_1] = value;
	                }
	            });
	            var req = new XMLHttpRequest();
	            var detectFunc = function () { };
	            req.onreadystatechange = detectFunc;
	            var result = req[SYMBOL_FAKE_ONREADYSTATECHANGE_1] === detectFunc;
	            req.onreadystatechange = null;
	            return result;
	        }
	    }
	    var globalEventHandlersEventNames = [
	        'abort',
	        'animationcancel',
	        'animationend',
	        'animationiteration',
	        'auxclick',
	        'beforeinput',
	        'blur',
	        'cancel',
	        'canplay',
	        'canplaythrough',
	        'change',
	        'compositionstart',
	        'compositionupdate',
	        'compositionend',
	        'cuechange',
	        'click',
	        'close',
	        'contextmenu',
	        'curechange',
	        'dblclick',
	        'drag',
	        'dragend',
	        'dragenter',
	        'dragexit',
	        'dragleave',
	        'dragover',
	        'drop',
	        'durationchange',
	        'emptied',
	        'ended',
	        'error',
	        'focus',
	        'focusin',
	        'focusout',
	        'gotpointercapture',
	        'input',
	        'invalid',
	        'keydown',
	        'keypress',
	        'keyup',
	        'load',
	        'loadstart',
	        'loadeddata',
	        'loadedmetadata',
	        'lostpointercapture',
	        'mousedown',
	        'mouseenter',
	        'mouseleave',
	        'mousemove',
	        'mouseout',
	        'mouseover',
	        'mouseup',
	        'mousewheel',
	        'orientationchange',
	        'pause',
	        'play',
	        'playing',
	        'pointercancel',
	        'pointerdown',
	        'pointerenter',
	        'pointerleave',
	        'pointerlockchange',
	        'mozpointerlockchange',
	        'webkitpointerlockerchange',
	        'pointerlockerror',
	        'mozpointerlockerror',
	        'webkitpointerlockerror',
	        'pointermove',
	        'pointout',
	        'pointerover',
	        'pointerup',
	        'progress',
	        'ratechange',
	        'reset',
	        'resize',
	        'scroll',
	        'seeked',
	        'seeking',
	        'select',
	        'selectionchange',
	        'selectstart',
	        'show',
	        'sort',
	        'stalled',
	        'submit',
	        'suspend',
	        'timeupdate',
	        'volumechange',
	        'touchcancel',
	        'touchmove',
	        'touchstart',
	        'touchend',
	        'transitioncancel',
	        'transitionend',
	        'waiting',
	        'wheel'
	    ];
	    var documentEventNames = [
	        'afterscriptexecute', 'beforescriptexecute', 'DOMContentLoaded', 'freeze', 'fullscreenchange',
	        'mozfullscreenchange', 'webkitfullscreenchange', 'msfullscreenchange', 'fullscreenerror',
	        'mozfullscreenerror', 'webkitfullscreenerror', 'msfullscreenerror', 'readystatechange',
	        'visibilitychange', 'resume'
	    ];
	    var windowEventNames = [
	        'absolutedeviceorientation',
	        'afterinput',
	        'afterprint',
	        'appinstalled',
	        'beforeinstallprompt',
	        'beforeprint',
	        'beforeunload',
	        'devicelight',
	        'devicemotion',
	        'deviceorientation',
	        'deviceorientationabsolute',
	        'deviceproximity',
	        'hashchange',
	        'languagechange',
	        'message',
	        'mozbeforepaint',
	        'offline',
	        'online',
	        'paint',
	        'pageshow',
	        'pagehide',
	        'popstate',
	        'rejectionhandled',
	        'storage',
	        'unhandledrejection',
	        'unload',
	        'userproximity',
	        'vrdisplayconnected',
	        'vrdisplaydisconnected',
	        'vrdisplaypresentchange'
	    ];
	    var htmlElementEventNames = [
	        'beforecopy', 'beforecut', 'beforepaste', 'copy', 'cut', 'paste', 'dragstart', 'loadend',
	        'animationstart', 'search', 'transitionrun', 'transitionstart', 'webkitanimationend',
	        'webkitanimationiteration', 'webkitanimationstart', 'webkittransitionend'
	    ];
	    var ieElementEventNames = [
	        'activate',
	        'afterupdate',
	        'ariarequest',
	        'beforeactivate',
	        'beforedeactivate',
	        'beforeeditfocus',
	        'beforeupdate',
	        'cellchange',
	        'controlselect',
	        'dataavailable',
	        'datasetchanged',
	        'datasetcomplete',
	        'errorupdate',
	        'filterchange',
	        'layoutcomplete',
	        'losecapture',
	        'move',
	        'moveend',
	        'movestart',
	        'propertychange',
	        'resizeend',
	        'resizestart',
	        'rowenter',
	        'rowexit',
	        'rowsdelete',
	        'rowsinserted',
	        'command',
	        'compassneedscalibration',
	        'deactivate',
	        'help',
	        'mscontentzoom',
	        'msmanipulationstatechanged',
	        'msgesturechange',
	        'msgesturedoubletap',
	        'msgestureend',
	        'msgesturehold',
	        'msgesturestart',
	        'msgesturetap',
	        'msgotpointercapture',
	        'msinertiastart',
	        'mslostpointercapture',
	        'mspointercancel',
	        'mspointerdown',
	        'mspointerenter',
	        'mspointerhover',
	        'mspointerleave',
	        'mspointermove',
	        'mspointerout',
	        'mspointerover',
	        'mspointerup',
	        'pointerout',
	        'mssitemodejumplistitemremoved',
	        'msthumbnailclick',
	        'stop',
	        'storagecommit'
	    ];
	    var webglEventNames = ['webglcontextrestored', 'webglcontextlost', 'webglcontextcreationerror'];
	    var formEventNames = ['autocomplete', 'autocompleteerror'];
	    var detailEventNames = ['toggle'];
	    var eventNames = __spreadArray(__spreadArray(__spreadArray(__spreadArray(__spreadArray(__spreadArray(__spreadArray(__spreadArray([], globalEventHandlersEventNames, true), webglEventNames, true), formEventNames, true), detailEventNames, true), documentEventNames, true), windowEventNames, true), htmlElementEventNames, true), ieElementEventNames, true);
	    // Whenever any eventListener fires, we check the eventListener target and all parents
	    // for `onwhatever` properties and replace them with zone-bound functions
	    // - Chrome (for now)
	    function patchViaCapturingAllTheEvents(api) {
	        var unboundKey = api.symbol('unbound');
	        var _loop_5 = function (i) {
	            var property = eventNames[i];
	            var onproperty = 'on' + property;
	            self.addEventListener(property, function (event) {
	                var elt = event.target, bound, source;
	                if (elt) {
	                    source = elt.constructor['name'] + '.' + onproperty;
	                }
	                else {
	                    source = 'unknown.' + onproperty;
	                }
	                while (elt) {
	                    if (elt[onproperty] && !elt[onproperty][unboundKey]) {
	                        bound = api.wrapWithCurrentZone(elt[onproperty], source);
	                        bound[unboundKey] = elt[onproperty];
	                        elt[onproperty] = bound;
	                    }
	                    elt = elt.parentElement;
	                }
	            }, true);
	        };
	        for (var i = 0; i < eventNames.length; i++) {
	            _loop_5(i);
	        }
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function registerElementPatch(_global, api) {
	        var _b = api.getGlobalObjects(), isBrowser = _b.isBrowser, isMix = _b.isMix;
	        if ((!isBrowser && !isMix) || !('registerElement' in _global.document)) {
	            return;
	        }
	        var callbacks = ['createdCallback', 'attachedCallback', 'detachedCallback', 'attributeChangedCallback'];
	        api.patchCallbacks(api, document, 'Document', 'registerElement', callbacks);
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    (function (_global) {
	        var symbolPrefix = _global['__Zone_symbol_prefix'] || '__zone_symbol__';
	        function __symbol__(name) {
	            return symbolPrefix + name;
	        }
	        _global[__symbol__('legacyPatch')] = function () {
	            var Zone = _global['Zone'];
	            Zone.__load_patch('defineProperty', function (global, Zone, api) {
	                api._redefineProperty = _redefineProperty;
	                propertyPatch();
	            });
	            Zone.__load_patch('registerElement', function (global, Zone, api) {
	                registerElementPatch(global, api);
	            });
	            Zone.__load_patch('EventTargetLegacy', function (global, Zone, api) {
	                eventTargetLegacyPatch(global, api);
	                propertyDescriptorLegacyPatch(api, global);
	            });
	        };
	    })(typeof window !== 'undefined' ?
	        window :
	        typeof commonjsGlobal !== 'undefined' ? commonjsGlobal : typeof self !== 'undefined' ? self : {});
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    var taskSymbol = zoneSymbol$1('zoneTask');
	    function patchTimer(window, setName, cancelName, nameSuffix) {
	        var setNative = null;
	        var clearNative = null;
	        setName += nameSuffix;
	        cancelName += nameSuffix;
	        var tasksByHandleId = {};
	        function scheduleTask(task) {
	            var data = task.data;
	            data.args[0] = function () {
	                return task.invoke.apply(this, arguments);
	            };
	            data.handleId = setNative.apply(window, data.args);
	            return task;
	        }
	        function clearTask(task) {
	            return clearNative.call(window, task.data.handleId);
	        }
	        setNative =
	            patchMethod(window, setName, function (delegate) { return function (self, args) {
	                if (typeof args[0] === 'function') {
	                    var options_1 = {
	                        isPeriodic: nameSuffix === 'Interval',
	                        delay: (nameSuffix === 'Timeout' || nameSuffix === 'Interval') ? args[1] || 0 :
	                            undefined,
	                        args: args
	                    };
	                    var callback_1 = args[0];
	                    args[0] = function timer() {
	                        try {
	                            return callback_1.apply(this, arguments);
	                        }
	                        finally {
	                            // issue-934, task will be cancelled
	                            // even it is a periodic task such as
	                            // setInterval
	                            // https://github.com/angular/angular/issues/40387
	                            // Cleanup tasksByHandleId should be handled before scheduleTask
	                            // Since some zoneSpec may intercept and doesn't trigger
	                            // scheduleFn(scheduleTask) provided here.
	                            if (!(options_1.isPeriodic)) {
	                                if (typeof options_1.handleId === 'number') {
	                                    // in non-nodejs env, we remove timerId
	                                    // from local cache
	                                    delete tasksByHandleId[options_1.handleId];
	                                }
	                                else if (options_1.handleId) {
	                                    // Node returns complex objects as handleIds
	                                    // we remove task reference from timer object
	                                    options_1.handleId[taskSymbol] = null;
	                                }
	                            }
	                        }
	                    };
	                    var task = scheduleMacroTaskWithCurrentZone(setName, args[0], options_1, scheduleTask, clearTask);
	                    if (!task) {
	                        return task;
	                    }
	                    // Node.js must additionally support the ref and unref functions.
	                    var handle = task.data.handleId;
	                    if (typeof handle === 'number') {
	                        // for non nodejs env, we save handleId: task
	                        // mapping in local cache for clearTimeout
	                        tasksByHandleId[handle] = task;
	                    }
	                    else if (handle) {
	                        // for nodejs env, we save task
	                        // reference in timerId Object for clearTimeout
	                        handle[taskSymbol] = task;
	                    }
	                    // check whether handle is null, because some polyfill or browser
	                    // may return undefined from setTimeout/setInterval/setImmediate/requestAnimationFrame
	                    if (handle && handle.ref && handle.unref && typeof handle.ref === 'function' &&
	                        typeof handle.unref === 'function') {
	                        task.ref = handle.ref.bind(handle);
	                        task.unref = handle.unref.bind(handle);
	                    }
	                    if (typeof handle === 'number' || handle) {
	                        return handle;
	                    }
	                    return task;
	                }
	                else {
	                    // cause an error by calling it directly.
	                    return delegate.apply(window, args);
	                }
	            }; });
	        clearNative =
	            patchMethod(window, cancelName, function (delegate) { return function (self, args) {
	                var id = args[0];
	                var task;
	                if (typeof id === 'number') {
	                    // non nodejs env.
	                    task = tasksByHandleId[id];
	                }
	                else {
	                    // nodejs env.
	                    task = id && id[taskSymbol];
	                    // other environments.
	                    if (!task) {
	                        task = id;
	                    }
	                }
	                if (task && typeof task.type === 'string') {
	                    if (task.state !== 'notScheduled' &&
	                        (task.cancelFn && task.data.isPeriodic || task.runCount === 0)) {
	                        if (typeof id === 'number') {
	                            delete tasksByHandleId[id];
	                        }
	                        else if (id) {
	                            id[taskSymbol] = null;
	                        }
	                        // Do not cancel already canceled functions
	                        task.zone.cancelTask(task);
	                    }
	                }
	                else {
	                    // cause an error by calling it directly.
	                    delegate.apply(window, args);
	                }
	            }; });
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function patchCustomElements(_global, api) {
	        var _b = api.getGlobalObjects(), isBrowser = _b.isBrowser, isMix = _b.isMix;
	        if ((!isBrowser && !isMix) || !_global['customElements'] || !('customElements' in _global)) {
	            return;
	        }
	        var callbacks = ['connectedCallback', 'disconnectedCallback', 'adoptedCallback', 'attributeChangedCallback'];
	        api.patchCallbacks(api, _global.customElements, 'customElements', 'define', callbacks);
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    function eventTargetPatch(_global, api) {
	        if (Zone[api.symbol('patchEventTarget')]) {
	            // EventTarget is already patched.
	            return;
	        }
	        var _b = api.getGlobalObjects(), eventNames = _b.eventNames, zoneSymbolEventNames = _b.zoneSymbolEventNames, TRUE_STR = _b.TRUE_STR, FALSE_STR = _b.FALSE_STR, ZONE_SYMBOL_PREFIX = _b.ZONE_SYMBOL_PREFIX;
	        //  predefine all __zone_symbol__ + eventName + true/false string
	        for (var i = 0; i < eventNames.length; i++) {
	            var eventName = eventNames[i];
	            var falseEventName = eventName + FALSE_STR;
	            var trueEventName = eventName + TRUE_STR;
	            var symbol = ZONE_SYMBOL_PREFIX + falseEventName;
	            var symbolCapture = ZONE_SYMBOL_PREFIX + trueEventName;
	            zoneSymbolEventNames[eventName] = {};
	            zoneSymbolEventNames[eventName][FALSE_STR] = symbol;
	            zoneSymbolEventNames[eventName][TRUE_STR] = symbolCapture;
	        }
	        var EVENT_TARGET = _global['EventTarget'];
	        if (!EVENT_TARGET || !EVENT_TARGET.prototype) {
	            return;
	        }
	        api.patchEventTarget(_global, api, [EVENT_TARGET && EVENT_TARGET.prototype]);
	        return true;
	    }
	    function patchEvent(global, api) {
	        api.patchEventPrototype(global, api);
	    }
	    /**
	     * @license
	     * Copyright Google LLC All Rights Reserved.
	     *
	     * Use of this source code is governed by an MIT-style license that can be
	     * found in the LICENSE file at https://angular.io/license
	     */
	    Zone.__load_patch('legacy', function (global) {
	        var legacyPatch = global[Zone.__symbol__('legacyPatch')];
	        if (legacyPatch) {
	            legacyPatch();
	        }
	    });
	    Zone.__load_patch('queueMicrotask', function (global, Zone, api) {
	        api.patchMethod(global, 'queueMicrotask', function (delegate) {
	            return function (self, args) {
	                Zone.current.scheduleMicroTask('queueMicrotask', args[0]);
	            };
	        });
	    });
	    Zone.__load_patch('timers', function (global) {
	        var set = 'set';
	        var clear = 'clear';
	        patchTimer(global, set, clear, 'Timeout');
	        patchTimer(global, set, clear, 'Interval');
	        patchTimer(global, set, clear, 'Immediate');
	    });
	    Zone.__load_patch('requestAnimationFrame', function (global) {
	        patchTimer(global, 'request', 'cancel', 'AnimationFrame');
	        patchTimer(global, 'mozRequest', 'mozCancel', 'AnimationFrame');
	        patchTimer(global, 'webkitRequest', 'webkitCancel', 'AnimationFrame');
	    });
	    Zone.__load_patch('blocking', function (global, Zone) {
	        var blockingMethods = ['alert', 'prompt', 'confirm'];
	        for (var i = 0; i < blockingMethods.length; i++) {
	            var name_2 = blockingMethods[i];
	            patchMethod(global, name_2, function (delegate, symbol, name) {
	                return function (s, args) {
	                    return Zone.current.run(delegate, global, args, name);
	                };
	            });
	        }
	    });
	    Zone.__load_patch('EventTarget', function (global, Zone, api) {
	        patchEvent(global, api);
	        eventTargetPatch(global, api);
	        // patch XMLHttpRequestEventTarget's addEventListener/removeEventListener
	        var XMLHttpRequestEventTarget = global['XMLHttpRequestEventTarget'];
	        if (XMLHttpRequestEventTarget && XMLHttpRequestEventTarget.prototype) {
	            api.patchEventTarget(global, api, [XMLHttpRequestEventTarget.prototype]);
	        }
	    });
	    Zone.__load_patch('MutationObserver', function (global, Zone, api) {
	        patchClass('MutationObserver');
	        patchClass('WebKitMutationObserver');
	    });
	    Zone.__load_patch('IntersectionObserver', function (global, Zone, api) {
	        patchClass('IntersectionObserver');
	    });
	    Zone.__load_patch('FileReader', function (global, Zone, api) {
	        patchClass('FileReader');
	    });
	    Zone.__load_patch('on_property', function (global, Zone, api) {
	        propertyDescriptorPatch(api, global);
	    });
	    Zone.__load_patch('customElements', function (global, Zone, api) {
	        patchCustomElements(global, api);
	    });
	    Zone.__load_patch('XHR', function (global, Zone) {
	        // Treat XMLHttpRequest as a macrotask.
	        patchXHR(global);
	        var XHR_TASK = zoneSymbol$1('xhrTask');
	        var XHR_SYNC = zoneSymbol$1('xhrSync');
	        var XHR_LISTENER = zoneSymbol$1('xhrListener');
	        var XHR_SCHEDULED = zoneSymbol$1('xhrScheduled');
	        var XHR_URL = zoneSymbol$1('xhrURL');
	        var XHR_ERROR_BEFORE_SCHEDULED = zoneSymbol$1('xhrErrorBeforeScheduled');
	        function patchXHR(window) {
	            var XMLHttpRequest = window['XMLHttpRequest'];
	            if (!XMLHttpRequest) {
	                // XMLHttpRequest is not available in service worker
	                return;
	            }
	            var XMLHttpRequestPrototype = XMLHttpRequest.prototype;
	            function findPendingTask(target) {
	                return target[XHR_TASK];
	            }
	            var oriAddListener = XMLHttpRequestPrototype[ZONE_SYMBOL_ADD_EVENT_LISTENER];
	            var oriRemoveListener = XMLHttpRequestPrototype[ZONE_SYMBOL_REMOVE_EVENT_LISTENER];
	            if (!oriAddListener) {
	                var XMLHttpRequestEventTarget_1 = window['XMLHttpRequestEventTarget'];
	                if (XMLHttpRequestEventTarget_1) {
	                    var XMLHttpRequestEventTargetPrototype = XMLHttpRequestEventTarget_1.prototype;
	                    oriAddListener = XMLHttpRequestEventTargetPrototype[ZONE_SYMBOL_ADD_EVENT_LISTENER];
	                    oriRemoveListener = XMLHttpRequestEventTargetPrototype[ZONE_SYMBOL_REMOVE_EVENT_LISTENER];
	                }
	            }
	            var READY_STATE_CHANGE = 'readystatechange';
	            var SCHEDULED = 'scheduled';
	            function scheduleTask(task) {
	                var data = task.data;
	                var target = data.target;
	                target[XHR_SCHEDULED] = false;
	                target[XHR_ERROR_BEFORE_SCHEDULED] = false;
	                // remove existing event listener
	                var listener = target[XHR_LISTENER];
	                if (!oriAddListener) {
	                    oriAddListener = target[ZONE_SYMBOL_ADD_EVENT_LISTENER];
	                    oriRemoveListener = target[ZONE_SYMBOL_REMOVE_EVENT_LISTENER];
	                }
	                if (listener) {
	                    oriRemoveListener.call(target, READY_STATE_CHANGE, listener);
	                }
	                var newListener = target[XHR_LISTENER] = function () {
	                    if (target.readyState === target.DONE) {
	                        // sometimes on some browsers XMLHttpRequest will fire onreadystatechange with
	                        // readyState=4 multiple times, so we need to check task state here
	                        if (!data.aborted && target[XHR_SCHEDULED] && task.state === SCHEDULED) {
	                            // check whether the xhr has registered onload listener
	                            // if that is the case, the task should invoke after all
	                            // onload listeners finish.
	                            // Also if the request failed without response (status = 0), the load event handler
	                            // will not be triggered, in that case, we should also invoke the placeholder callback
	                            // to close the XMLHttpRequest::send macroTask.
	                            // https://github.com/angular/angular/issues/38795
	                            var loadTasks = target[Zone.__symbol__('loadfalse')];
	                            if (target.status !== 0 && loadTasks && loadTasks.length > 0) {
	                                var oriInvoke_1 = task.invoke;
	                                task.invoke = function () {
	                                    // need to load the tasks again, because in other
	                                    // load listener, they may remove themselves
	                                    var loadTasks = target[Zone.__symbol__('loadfalse')];
	                                    for (var i = 0; i < loadTasks.length; i++) {
	                                        if (loadTasks[i] === task) {
	                                            loadTasks.splice(i, 1);
	                                        }
	                                    }
	                                    if (!data.aborted && task.state === SCHEDULED) {
	                                        oriInvoke_1.call(task);
	                                    }
	                                };
	                                loadTasks.push(task);
	                            }
	                            else {
	                                task.invoke();
	                            }
	                        }
	                        else if (!data.aborted && target[XHR_SCHEDULED] === false) {
	                            // error occurs when xhr.send()
	                            target[XHR_ERROR_BEFORE_SCHEDULED] = true;
	                        }
	                    }
	                };
	                oriAddListener.call(target, READY_STATE_CHANGE, newListener);
	                var storedTask = target[XHR_TASK];
	                if (!storedTask) {
	                    target[XHR_TASK] = task;
	                }
	                sendNative.apply(target, data.args);
	                target[XHR_SCHEDULED] = true;
	                return task;
	            }
	            function placeholderCallback() { }
	            function clearTask(task) {
	                var data = task.data;
	                // Note - ideally, we would call data.target.removeEventListener here, but it's too late
	                // to prevent it from firing. So instead, we store info for the event listener.
	                data.aborted = true;
	                return abortNative.apply(data.target, data.args);
	            }
	            var openNative = patchMethod(XMLHttpRequestPrototype, 'open', function () { return function (self, args) {
	                self[XHR_SYNC] = args[2] == false;
	                self[XHR_URL] = args[1];
	                return openNative.apply(self, args);
	            }; });
	            var XMLHTTPREQUEST_SOURCE = 'XMLHttpRequest.send';
	            var fetchTaskAborting = zoneSymbol$1('fetchTaskAborting');
	            var fetchTaskScheduling = zoneSymbol$1('fetchTaskScheduling');
	            var sendNative = patchMethod(XMLHttpRequestPrototype, 'send', function () { return function (self, args) {
	                if (Zone.current[fetchTaskScheduling] === true) {
	                    // a fetch is scheduling, so we are using xhr to polyfill fetch
	                    // and because we already schedule macroTask for fetch, we should
	                    // not schedule a macroTask for xhr again
	                    return sendNative.apply(self, args);
	                }
	                if (self[XHR_SYNC]) {
	                    // if the XHR is sync there is no task to schedule, just execute the code.
	                    return sendNative.apply(self, args);
	                }
	                else {
	                    var options = { target: self, url: self[XHR_URL], isPeriodic: false, args: args, aborted: false };
	                    var task = scheduleMacroTaskWithCurrentZone(XMLHTTPREQUEST_SOURCE, placeholderCallback, options, scheduleTask, clearTask);
	                    if (self && self[XHR_ERROR_BEFORE_SCHEDULED] === true && !options.aborted &&
	                        task.state === SCHEDULED) {
	                        // xhr request throw error when send
	                        // we should invoke task instead of leaving a scheduled
	                        // pending macroTask
	                        task.invoke();
	                    }
	                }
	            }; });
	            var abortNative = patchMethod(XMLHttpRequestPrototype, 'abort', function () { return function (self, args) {
	                var task = findPendingTask(self);
	                if (task && typeof task.type == 'string') {
	                    // If the XHR has already completed, do nothing.
	                    // If the XHR has already been aborted, do nothing.
	                    // Fix #569, call abort multiple times before done will cause
	                    // macroTask task count be negative number
	                    if (task.cancelFn == null || (task.data && task.data.aborted)) {
	                        return;
	                    }
	                    task.zone.cancelTask(task);
	                }
	                else if (Zone.current[fetchTaskAborting] === true) {
	                    // the abort is called from fetch polyfill, we need to call native abort of XHR.
	                    return abortNative.apply(self, args);
	                }
	                // Otherwise, we are trying to abort an XHR which has not yet been sent, so there is no
	                // task
	                // to cancel. Do nothing.
	            }; });
	        }
	    });
	    Zone.__load_patch('geolocation', function (global) {
	        /// GEO_LOCATION
	        if (global['navigator'] && global['navigator'].geolocation) {
	            patchPrototype(global['navigator'].geolocation, ['getCurrentPosition', 'watchPosition']);
	        }
	    });
	    Zone.__load_patch('PromiseRejectionEvent', function (global, Zone) {
	        // handle unhandled promise rejection
	        function findPromiseRejectionHandler(evtName) {
	            return function (e) {
	                var eventTasks = findEventTasks(global, evtName);
	                eventTasks.forEach(function (eventTask) {
	                    // windows has added unhandledrejection event listener
	                    // trigger the event listener
	                    var PromiseRejectionEvent = global['PromiseRejectionEvent'];
	                    if (PromiseRejectionEvent) {
	                        var evt = new PromiseRejectionEvent(evtName, { promise: e.promise, reason: e.rejection });
	                        eventTask.invoke(evt);
	                    }
	                });
	            };
	        }
	        if (global['PromiseRejectionEvent']) {
	            Zone[zoneSymbol$1('unhandledPromiseRejectionHandler')] =
	                findPromiseRejectionHandler('unhandledrejection');
	            Zone[zoneSymbol$1('rejectionHandledHandler')] =
	                findPromiseRejectionHandler('rejectionhandled');
	        }
	    });
	}));

	return zone;

})();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiem9uZS5qcyIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vZXh0ZXJuYWwvbnBtL25vZGVfbW9kdWxlcy96b25lLmpzL2Rpc3Qvem9uZS5qcyJdLCJzb3VyY2VzQ29udGVudCI6WyIndXNlIHN0cmljdCc7XG52YXIgX19zcHJlYWRBcnJheSA9ICh0aGlzICYmIHRoaXMuX19zcHJlYWRBcnJheSkgfHwgZnVuY3Rpb24gKHRvLCBmcm9tLCBwYWNrKSB7XG4gICAgaWYgKHBhY2sgfHwgYXJndW1lbnRzLmxlbmd0aCA9PT0gMikgZm9yICh2YXIgaSA9IDAsIGwgPSBmcm9tLmxlbmd0aCwgYXI7IGkgPCBsOyBpKyspIHtcbiAgICAgICAgaWYgKGFyIHx8ICEoaSBpbiBmcm9tKSkge1xuICAgICAgICAgICAgaWYgKCFhcikgYXIgPSBBcnJheS5wcm90b3R5cGUuc2xpY2UuY2FsbChmcm9tLCAwLCBpKTtcbiAgICAgICAgICAgIGFyW2ldID0gZnJvbVtpXTtcbiAgICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gdG8uY29uY2F0KGFyIHx8IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGZyb20pKTtcbn07XG4vKipcbiAqIEBsaWNlbnNlIEFuZ3VsYXIgdjE1LjEuMC1uZXh0LjBcbiAqIChjKSAyMDEwLTIwMjIgR29vZ2xlIExMQy4gaHR0cHM6Ly9hbmd1bGFyLmlvL1xuICogTGljZW5zZTogTUlUXG4gKi9cbihmdW5jdGlvbiAoZmFjdG9yeSkge1xuICAgIHR5cGVvZiBkZWZpbmUgPT09ICdmdW5jdGlvbicgJiYgZGVmaW5lLmFtZCA/IGRlZmluZShmYWN0b3J5KSA6XG4gICAgICAgIGZhY3RvcnkoKTtcbn0pKChmdW5jdGlvbiAoKSB7XG4gICAgJ3VzZSBzdHJpY3QnO1xuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgICgoZnVuY3Rpb24gKGdsb2JhbCkge1xuICAgICAgICB2YXIgcGVyZm9ybWFuY2UgPSBnbG9iYWxbJ3BlcmZvcm1hbmNlJ107XG4gICAgICAgIGZ1bmN0aW9uIG1hcmsobmFtZSkge1xuICAgICAgICAgICAgcGVyZm9ybWFuY2UgJiYgcGVyZm9ybWFuY2VbJ21hcmsnXSAmJiBwZXJmb3JtYW5jZVsnbWFyayddKG5hbWUpO1xuICAgICAgICB9XG4gICAgICAgIGZ1bmN0aW9uIHBlcmZvcm1hbmNlTWVhc3VyZShuYW1lLCBsYWJlbCkge1xuICAgICAgICAgICAgcGVyZm9ybWFuY2UgJiYgcGVyZm9ybWFuY2VbJ21lYXN1cmUnXSAmJiBwZXJmb3JtYW5jZVsnbWVhc3VyZSddKG5hbWUsIGxhYmVsKTtcbiAgICAgICAgfVxuICAgICAgICBtYXJrKCdab25lJyk7XG4gICAgICAgIC8vIEluaXRpYWxpemUgYmVmb3JlIGl0J3MgYWNjZXNzZWQgYmVsb3cuXG4gICAgICAgIC8vIF9fWm9uZV9zeW1ib2xfcHJlZml4IGdsb2JhbCBjYW4gYmUgdXNlZCB0byBvdmVycmlkZSB0aGUgZGVmYXVsdCB6b25lXG4gICAgICAgIC8vIHN5bWJvbCBwcmVmaXggd2l0aCBhIGN1c3RvbSBvbmUgaWYgbmVlZGVkLlxuICAgICAgICB2YXIgc3ltYm9sUHJlZml4ID0gZ2xvYmFsWydfX1pvbmVfc3ltYm9sX3ByZWZpeCddIHx8ICdfX3pvbmVfc3ltYm9sX18nO1xuICAgICAgICBmdW5jdGlvbiBfX3N5bWJvbF9fKG5hbWUpIHtcbiAgICAgICAgICAgIHJldHVybiBzeW1ib2xQcmVmaXggKyBuYW1lO1xuICAgICAgICB9XG4gICAgICAgIHZhciBjaGVja0R1cGxpY2F0ZSA9IGdsb2JhbFtfX3N5bWJvbF9fKCdmb3JjZUR1cGxpY2F0ZVpvbmVDaGVjaycpXSA9PT0gdHJ1ZTtcbiAgICAgICAgaWYgKGdsb2JhbFsnWm9uZSddKSB7XG4gICAgICAgICAgICAvLyBpZiBnbG9iYWxbJ1pvbmUnXSBhbHJlYWR5IGV4aXN0cyAobWF5YmUgem9uZS5qcyB3YXMgYWxyZWFkeSBsb2FkZWQgb3JcbiAgICAgICAgICAgIC8vIHNvbWUgb3RoZXIgbGliIGFsc28gcmVnaXN0ZXJlZCBhIGdsb2JhbCBvYmplY3QgbmFtZWQgWm9uZSksIHdlIG1heSBuZWVkXG4gICAgICAgICAgICAvLyB0byB0aHJvdyBhbiBlcnJvciwgYnV0IHNvbWV0aW1lcyB1c2VyIG1heSBub3Qgd2FudCB0aGlzIGVycm9yLlxuICAgICAgICAgICAgLy8gRm9yIGV4YW1wbGUsXG4gICAgICAgICAgICAvLyB3ZSBoYXZlIHR3byB3ZWIgcGFnZXMsIHBhZ2UxIGluY2x1ZGVzIHpvbmUuanMsIHBhZ2UyIGRvZXNuJ3QuXG4gICAgICAgICAgICAvLyBhbmQgdGhlIDFzdCB0aW1lIHVzZXIgbG9hZCBwYWdlMSBhbmQgcGFnZTIsIGV2ZXJ5dGhpbmcgd29yayBmaW5lLFxuICAgICAgICAgICAgLy8gYnV0IHdoZW4gdXNlciBsb2FkIHBhZ2UyIGFnYWluLCBlcnJvciBvY2N1cnMgYmVjYXVzZSBnbG9iYWxbJ1pvbmUnXSBhbHJlYWR5IGV4aXN0cy5cbiAgICAgICAgICAgIC8vIHNvIHdlIGFkZCBhIGZsYWcgdG8gbGV0IHVzZXIgY2hvb3NlIHdoZXRoZXIgdG8gdGhyb3cgdGhpcyBlcnJvciBvciBub3QuXG4gICAgICAgICAgICAvLyBCeSBkZWZhdWx0LCBpZiBleGlzdGluZyBab25lIGlzIGZyb20gem9uZS5qcywgd2Ugd2lsbCBub3QgdGhyb3cgdGhlIGVycm9yLlxuICAgICAgICAgICAgaWYgKGNoZWNrRHVwbGljYXRlIHx8IHR5cGVvZiBnbG9iYWxbJ1pvbmUnXS5fX3N5bWJvbF9fICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdab25lIGFscmVhZHkgbG9hZGVkLicpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGdsb2JhbFsnWm9uZSddO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHZhciBab25lID0gLyoqIEBjbGFzcyAqLyAoZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgZnVuY3Rpb24gWm9uZShwYXJlbnQsIHpvbmVTcGVjKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5fcGFyZW50ID0gcGFyZW50O1xuICAgICAgICAgICAgICAgIHRoaXMuX25hbWUgPSB6b25lU3BlYyA/IHpvbmVTcGVjLm5hbWUgfHwgJ3VubmFtZWQnIDogJzxyb290Pic7XG4gICAgICAgICAgICAgICAgdGhpcy5fcHJvcGVydGllcyA9IHpvbmVTcGVjICYmIHpvbmVTcGVjLnByb3BlcnRpZXMgfHwge307XG4gICAgICAgICAgICAgICAgdGhpcy5fem9uZURlbGVnYXRlID1cbiAgICAgICAgICAgICAgICAgICAgbmV3IF9ab25lRGVsZWdhdGUodGhpcywgdGhpcy5fcGFyZW50ICYmIHRoaXMuX3BhcmVudC5fem9uZURlbGVnYXRlLCB6b25lU3BlYyk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBab25lLmFzc2VydFpvbmVQYXRjaGVkID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIGlmIChnbG9iYWxbJ1Byb21pc2UnXSAhPT0gcGF0Y2hlc1snWm9uZUF3YXJlUHJvbWlzZSddKSB7XG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignWm9uZS5qcyBoYXMgZGV0ZWN0ZWQgdGhhdCBab25lQXdhcmVQcm9taXNlIGAod2luZG93fGdsb2JhbCkuUHJvbWlzZWAgJyArXG4gICAgICAgICAgICAgICAgICAgICAgICAnaGFzIGJlZW4gb3ZlcndyaXR0ZW4uXFxuJyArXG4gICAgICAgICAgICAgICAgICAgICAgICAnTW9zdCBsaWtlbHkgY2F1c2UgaXMgdGhhdCBhIFByb21pc2UgcG9seWZpbGwgaGFzIGJlZW4gbG9hZGVkICcgK1xuICAgICAgICAgICAgICAgICAgICAgICAgJ2FmdGVyIFpvbmUuanMgKFBvbHlmaWxsaW5nIFByb21pc2UgYXBpIGlzIG5vdCBuZWNlc3Nhcnkgd2hlbiB6b25lLmpzIGlzIGxvYWRlZC4gJyArXG4gICAgICAgICAgICAgICAgICAgICAgICAnSWYgeW91IG11c3QgbG9hZCBvbmUsIGRvIHNvIGJlZm9yZSBsb2FkaW5nIHpvbmUuanMuKScpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoWm9uZSwgXCJyb290XCIsIHtcbiAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHpvbmUgPSBab25lLmN1cnJlbnQ7XG4gICAgICAgICAgICAgICAgICAgIHdoaWxlICh6b25lLnBhcmVudCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgem9uZSA9IHpvbmUucGFyZW50O1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB6b25lO1xuICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgZW51bWVyYWJsZTogZmFsc2UsXG4gICAgICAgICAgICAgICAgY29uZmlndXJhYmxlOiB0cnVlXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShab25lLCBcImN1cnJlbnRcIiwge1xuICAgICAgICAgICAgICAgIGdldDogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gX2N1cnJlbnRab25lRnJhbWUuem9uZTtcbiAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgIGVudW1lcmFibGU6IGZhbHNlLFxuICAgICAgICAgICAgICAgIGNvbmZpZ3VyYWJsZTogdHJ1ZVxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoWm9uZSwgXCJjdXJyZW50VGFza1wiLCB7XG4gICAgICAgICAgICAgICAgZ2V0OiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBfY3VycmVudFRhc2s7XG4gICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICBlbnVtZXJhYmxlOiBmYWxzZSxcbiAgICAgICAgICAgICAgICBjb25maWd1cmFibGU6IHRydWVcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOnJlcXVpcmUtaW50ZXJuYWwtd2l0aC11bmRlcnNjb3JlXG4gICAgICAgICAgICBab25lLl9fbG9hZF9wYXRjaCA9IGZ1bmN0aW9uIChuYW1lLCBmbiwgaWdub3JlRHVwbGljYXRlKSB7XG4gICAgICAgICAgICAgICAgaWYgKGlnbm9yZUR1cGxpY2F0ZSA9PT0gdm9pZCAwKSB7IGlnbm9yZUR1cGxpY2F0ZSA9IGZhbHNlOyB9XG4gICAgICAgICAgICAgICAgaWYgKHBhdGNoZXMuaGFzT3duUHJvcGVydHkobmFtZSkpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gYGNoZWNrRHVwbGljYXRlYCBvcHRpb24gaXMgZGVmaW5lZCBmcm9tIGdsb2JhbCB2YXJpYWJsZVxuICAgICAgICAgICAgICAgICAgICAvLyBzbyBpdCB3b3JrcyBmb3IgYWxsIG1vZHVsZXMuXG4gICAgICAgICAgICAgICAgICAgIC8vIGBpZ25vcmVEdXBsaWNhdGVgIGNhbiB3b3JrIGZvciB0aGUgc3BlY2lmaWVkIG1vZHVsZVxuICAgICAgICAgICAgICAgICAgICBpZiAoIWlnbm9yZUR1cGxpY2F0ZSAmJiBjaGVja0R1cGxpY2F0ZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgRXJyb3IoJ0FscmVhZHkgbG9hZGVkIHBhdGNoOiAnICsgbmFtZSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSBpZiAoIWdsb2JhbFsnX19ab25lX2Rpc2FibGVfJyArIG5hbWVdKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciBwZXJmTmFtZSA9ICdab25lOicgKyBuYW1lO1xuICAgICAgICAgICAgICAgICAgICBtYXJrKHBlcmZOYW1lKTtcbiAgICAgICAgICAgICAgICAgICAgcGF0Y2hlc1tuYW1lXSA9IGZuKGdsb2JhbCwgWm9uZSwgX2FwaSk7XG4gICAgICAgICAgICAgICAgICAgIHBlcmZvcm1hbmNlTWVhc3VyZShwZXJmTmFtZSwgcGVyZk5hbWUpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoWm9uZS5wcm90b3R5cGUsIFwicGFyZW50XCIsIHtcbiAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3BhcmVudDtcbiAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgIGVudW1lcmFibGU6IGZhbHNlLFxuICAgICAgICAgICAgICAgIGNvbmZpZ3VyYWJsZTogdHJ1ZVxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoWm9uZS5wcm90b3R5cGUsIFwibmFtZVwiLCB7XG4gICAgICAgICAgICAgICAgZ2V0OiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9uYW1lO1xuICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgZW51bWVyYWJsZTogZmFsc2UsXG4gICAgICAgICAgICAgICAgY29uZmlndXJhYmxlOiB0cnVlXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIFpvbmUucHJvdG90eXBlLmdldCA9IGZ1bmN0aW9uIChrZXkpIHtcbiAgICAgICAgICAgICAgICB2YXIgem9uZSA9IHRoaXMuZ2V0Wm9uZVdpdGgoa2V5KTtcbiAgICAgICAgICAgICAgICBpZiAoem9uZSlcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHpvbmUuX3Byb3BlcnRpZXNba2V5XTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5nZXRab25lV2l0aCA9IGZ1bmN0aW9uIChrZXkpIHtcbiAgICAgICAgICAgICAgICB2YXIgY3VycmVudCA9IHRoaXM7XG4gICAgICAgICAgICAgICAgd2hpbGUgKGN1cnJlbnQpIHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKGN1cnJlbnQuX3Byb3BlcnRpZXMuaGFzT3duUHJvcGVydHkoa2V5KSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGN1cnJlbnQ7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgY3VycmVudCA9IGN1cnJlbnQuX3BhcmVudDtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgWm9uZS5wcm90b3R5cGUuZm9yayA9IGZ1bmN0aW9uICh6b25lU3BlYykge1xuICAgICAgICAgICAgICAgIGlmICghem9uZVNwZWMpXG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignWm9uZVNwZWMgcmVxdWlyZWQhJyk7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3pvbmVEZWxlZ2F0ZS5mb3JrKHRoaXMsIHpvbmVTcGVjKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS53cmFwID0gZnVuY3Rpb24gKGNhbGxiYWNrLCBzb3VyY2UpIHtcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGNhbGxiYWNrICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignRXhwZWN0aW5nIGZ1bmN0aW9uIGdvdDogJyArIGNhbGxiYWNrKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdmFyIF9jYWxsYmFjayA9IHRoaXMuX3pvbmVEZWxlZ2F0ZS5pbnRlcmNlcHQodGhpcywgY2FsbGJhY2ssIHNvdXJjZSk7XG4gICAgICAgICAgICAgICAgdmFyIHpvbmUgPSB0aGlzO1xuICAgICAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB6b25lLnJ1bkd1YXJkZWQoX2NhbGxiYWNrLCB0aGlzLCBhcmd1bWVudHMsIHNvdXJjZSk7XG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5ydW4gPSBmdW5jdGlvbiAoY2FsbGJhY2ssIGFwcGx5VGhpcywgYXBwbHlBcmdzLCBzb3VyY2UpIHtcbiAgICAgICAgICAgICAgICBfY3VycmVudFpvbmVGcmFtZSA9IHsgcGFyZW50OiBfY3VycmVudFpvbmVGcmFtZSwgem9uZTogdGhpcyB9O1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl96b25lRGVsZWdhdGUuaW52b2tlKHRoaXMsIGNhbGxiYWNrLCBhcHBseVRoaXMsIGFwcGx5QXJncywgc291cmNlKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZmluYWxseSB7XG4gICAgICAgICAgICAgICAgICAgIF9jdXJyZW50Wm9uZUZyYW1lID0gX2N1cnJlbnRab25lRnJhbWUucGFyZW50O1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5ydW5HdWFyZGVkID0gZnVuY3Rpb24gKGNhbGxiYWNrLCBhcHBseVRoaXMsIGFwcGx5QXJncywgc291cmNlKSB7XG4gICAgICAgICAgICAgICAgaWYgKGFwcGx5VGhpcyA9PT0gdm9pZCAwKSB7IGFwcGx5VGhpcyA9IG51bGw7IH1cbiAgICAgICAgICAgICAgICBfY3VycmVudFpvbmVGcmFtZSA9IHsgcGFyZW50OiBfY3VycmVudFpvbmVGcmFtZSwgem9uZTogdGhpcyB9O1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdGhpcy5fem9uZURlbGVnYXRlLmludm9rZSh0aGlzLCBjYWxsYmFjaywgYXBwbHlUaGlzLCBhcHBseUFyZ3MsIHNvdXJjZSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodGhpcy5fem9uZURlbGVnYXRlLmhhbmRsZUVycm9yKHRoaXMsIGVycm9yKSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGZpbmFsbHkge1xuICAgICAgICAgICAgICAgICAgICBfY3VycmVudFpvbmVGcmFtZSA9IF9jdXJyZW50Wm9uZUZyYW1lLnBhcmVudDtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgWm9uZS5wcm90b3R5cGUucnVuVGFzayA9IGZ1bmN0aW9uICh0YXNrLCBhcHBseVRoaXMsIGFwcGx5QXJncykge1xuICAgICAgICAgICAgICAgIGlmICh0YXNrLnpvbmUgIT0gdGhpcykge1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0EgdGFzayBjYW4gb25seSBiZSBydW4gaW4gdGhlIHpvbmUgb2YgY3JlYXRpb24hIChDcmVhdGlvbjogJyArXG4gICAgICAgICAgICAgICAgICAgICAgICAodGFzay56b25lIHx8IE5PX1pPTkUpLm5hbWUgKyAnOyBFeGVjdXRpb246ICcgKyB0aGlzLm5hbWUgKyAnKScpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAvLyBodHRwczovL2dpdGh1Yi5jb20vYW5ndWxhci96b25lLmpzL2lzc3Vlcy83NzgsIHNvbWV0aW1lcyBldmVudFRhc2tcbiAgICAgICAgICAgICAgICAvLyB3aWxsIHJ1biBpbiBub3RTY2hlZHVsZWQoY2FuY2VsZWQpIHN0YXRlLCB3ZSBzaG91bGQgbm90IHRyeSB0b1xuICAgICAgICAgICAgICAgIC8vIHJ1biBzdWNoIGtpbmQgb2YgdGFzayBidXQganVzdCByZXR1cm5cbiAgICAgICAgICAgICAgICBpZiAodGFzay5zdGF0ZSA9PT0gbm90U2NoZWR1bGVkICYmICh0YXNrLnR5cGUgPT09IGV2ZW50VGFzayB8fCB0YXNrLnR5cGUgPT09IG1hY3JvVGFzaykpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB2YXIgcmVFbnRyeUd1YXJkID0gdGFzay5zdGF0ZSAhPSBydW5uaW5nO1xuICAgICAgICAgICAgICAgIHJlRW50cnlHdWFyZCAmJiB0YXNrLl90cmFuc2l0aW9uVG8ocnVubmluZywgc2NoZWR1bGVkKTtcbiAgICAgICAgICAgICAgICB0YXNrLnJ1bkNvdW50Kys7XG4gICAgICAgICAgICAgICAgdmFyIHByZXZpb3VzVGFzayA9IF9jdXJyZW50VGFzaztcbiAgICAgICAgICAgICAgICBfY3VycmVudFRhc2sgPSB0YXNrO1xuICAgICAgICAgICAgICAgIF9jdXJyZW50Wm9uZUZyYW1lID0geyBwYXJlbnQ6IF9jdXJyZW50Wm9uZUZyYW1lLCB6b25lOiB0aGlzIH07XG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRhc2sudHlwZSA9PSBtYWNyb1Rhc2sgJiYgdGFzay5kYXRhICYmICF0YXNrLmRhdGEuaXNQZXJpb2RpYykge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGFzay5jYW5jZWxGbiA9IHVuZGVmaW5lZDtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3pvbmVEZWxlZ2F0ZS5pbnZva2VUYXNrKHRoaXMsIHRhc2ssIGFwcGx5VGhpcywgYXBwbHlBcmdzKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0aGlzLl96b25lRGVsZWdhdGUuaGFuZGxlRXJyb3IodGhpcywgZXJyb3IpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZmluYWxseSB7XG4gICAgICAgICAgICAgICAgICAgIC8vIGlmIHRoZSB0YXNrJ3Mgc3RhdGUgaXMgbm90U2NoZWR1bGVkIG9yIHVua25vd24sIHRoZW4gaXQgaGFzIGFscmVhZHkgYmVlbiBjYW5jZWxsZWRcbiAgICAgICAgICAgICAgICAgICAgLy8gd2Ugc2hvdWxkIG5vdCByZXNldCB0aGUgc3RhdGUgdG8gc2NoZWR1bGVkXG4gICAgICAgICAgICAgICAgICAgIGlmICh0YXNrLnN0YXRlICE9PSBub3RTY2hlZHVsZWQgJiYgdGFzay5zdGF0ZSAhPT0gdW5rbm93bikge1xuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHRhc2sudHlwZSA9PSBldmVudFRhc2sgfHwgKHRhc2suZGF0YSAmJiB0YXNrLmRhdGEuaXNQZXJpb2RpYykpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZUVudHJ5R3VhcmQgJiYgdGFzay5fdHJhbnNpdGlvblRvKHNjaGVkdWxlZCwgcnVubmluZyk7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YXNrLnJ1bkNvdW50ID0gMDtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl91cGRhdGVUYXNrQ291bnQodGFzaywgLTEpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlRW50cnlHdWFyZCAmJlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YXNrLl90cmFuc2l0aW9uVG8obm90U2NoZWR1bGVkLCBydW5uaW5nLCBub3RTY2hlZHVsZWQpO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIF9jdXJyZW50Wm9uZUZyYW1lID0gX2N1cnJlbnRab25lRnJhbWUucGFyZW50O1xuICAgICAgICAgICAgICAgICAgICBfY3VycmVudFRhc2sgPSBwcmV2aW91c1Rhc2s7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmUucHJvdG90eXBlLnNjaGVkdWxlVGFzayA9IGZ1bmN0aW9uICh0YXNrKSB7XG4gICAgICAgICAgICAgICAgaWYgKHRhc2suem9uZSAmJiB0YXNrLnpvbmUgIT09IHRoaXMpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gY2hlY2sgaWYgdGhlIHRhc2sgd2FzIHJlc2NoZWR1bGVkLCB0aGUgbmV3Wm9uZVxuICAgICAgICAgICAgICAgICAgICAvLyBzaG91bGQgbm90IGJlIHRoZSBjaGlsZHJlbiBvZiB0aGUgb3JpZ2luYWwgem9uZVxuICAgICAgICAgICAgICAgICAgICB2YXIgbmV3Wm9uZSA9IHRoaXM7XG4gICAgICAgICAgICAgICAgICAgIHdoaWxlIChuZXdab25lKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAobmV3Wm9uZSA9PT0gdGFzay56b25lKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgRXJyb3IoXCJjYW4gbm90IHJlc2NoZWR1bGUgdGFzayB0byBcIi5jb25jYXQodGhpcy5uYW1lLCBcIiB3aGljaCBpcyBkZXNjZW5kYW50cyBvZiB0aGUgb3JpZ2luYWwgem9uZSBcIikuY29uY2F0KHRhc2suem9uZS5uYW1lKSk7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICBuZXdab25lID0gbmV3Wm9uZS5wYXJlbnQ7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdGFzay5fdHJhbnNpdGlvblRvKHNjaGVkdWxpbmcsIG5vdFNjaGVkdWxlZCk7XG4gICAgICAgICAgICAgICAgdmFyIHpvbmVEZWxlZ2F0ZXMgPSBbXTtcbiAgICAgICAgICAgICAgICB0YXNrLl96b25lRGVsZWdhdGVzID0gem9uZURlbGVnYXRlcztcbiAgICAgICAgICAgICAgICB0YXNrLl96b25lID0gdGhpcztcbiAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICB0YXNrID0gdGhpcy5fem9uZURlbGVnYXRlLnNjaGVkdWxlVGFzayh0aGlzLCB0YXNrKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgICAgICAgICAvLyBzaG91bGQgc2V0IHRhc2sncyBzdGF0ZSB0byB1bmtub3duIHdoZW4gc2NoZWR1bGVUYXNrIHRocm93IGVycm9yXG4gICAgICAgICAgICAgICAgICAgIC8vIGJlY2F1c2UgdGhlIGVyciBtYXkgZnJvbSByZXNjaGVkdWxlLCBzbyB0aGUgZnJvbVN0YXRlIG1heWJlIG5vdFNjaGVkdWxlZFxuICAgICAgICAgICAgICAgICAgICB0YXNrLl90cmFuc2l0aW9uVG8odW5rbm93biwgc2NoZWR1bGluZywgbm90U2NoZWR1bGVkKTtcbiAgICAgICAgICAgICAgICAgICAgLy8gVE9ETzogQEppYUxpUGFzc2lvbiwgc2hvdWxkIHdlIGNoZWNrIHRoZSByZXN1bHQgZnJvbSBoYW5kbGVFcnJvcj9cbiAgICAgICAgICAgICAgICAgICAgdGhpcy5fem9uZURlbGVnYXRlLmhhbmRsZUVycm9yKHRoaXMsIGVycik7XG4gICAgICAgICAgICAgICAgICAgIHRocm93IGVycjtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKHRhc2suX3pvbmVEZWxlZ2F0ZXMgPT09IHpvbmVEZWxlZ2F0ZXMpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gd2UgaGF2ZSB0byBjaGVjayBiZWNhdXNlIGludGVybmFsbHkgdGhlIGRlbGVnYXRlIGNhbiByZXNjaGVkdWxlIHRoZSB0YXNrLlxuICAgICAgICAgICAgICAgICAgICB0aGlzLl91cGRhdGVUYXNrQ291bnQodGFzaywgMSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICh0YXNrLnN0YXRlID09IHNjaGVkdWxpbmcpIHtcbiAgICAgICAgICAgICAgICAgICAgdGFzay5fdHJhbnNpdGlvblRvKHNjaGVkdWxlZCwgc2NoZWR1bGluZyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiB0YXNrO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmUucHJvdG90eXBlLnNjaGVkdWxlTWljcm9UYXNrID0gZnVuY3Rpb24gKHNvdXJjZSwgY2FsbGJhY2ssIGRhdGEsIGN1c3RvbVNjaGVkdWxlKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuc2NoZWR1bGVUYXNrKG5ldyBab25lVGFzayhtaWNyb1Rhc2ssIHNvdXJjZSwgY2FsbGJhY2ssIGRhdGEsIGN1c3RvbVNjaGVkdWxlLCB1bmRlZmluZWQpKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5zY2hlZHVsZU1hY3JvVGFzayA9IGZ1bmN0aW9uIChzb3VyY2UsIGNhbGxiYWNrLCBkYXRhLCBjdXN0b21TY2hlZHVsZSwgY3VzdG9tQ2FuY2VsKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuc2NoZWR1bGVUYXNrKG5ldyBab25lVGFzayhtYWNyb1Rhc2ssIHNvdXJjZSwgY2FsbGJhY2ssIGRhdGEsIGN1c3RvbVNjaGVkdWxlLCBjdXN0b21DYW5jZWwpKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5zY2hlZHVsZUV2ZW50VGFzayA9IGZ1bmN0aW9uIChzb3VyY2UsIGNhbGxiYWNrLCBkYXRhLCBjdXN0b21TY2hlZHVsZSwgY3VzdG9tQ2FuY2VsKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuc2NoZWR1bGVUYXNrKG5ldyBab25lVGFzayhldmVudFRhc2ssIHNvdXJjZSwgY2FsbGJhY2ssIGRhdGEsIGN1c3RvbVNjaGVkdWxlLCBjdXN0b21DYW5jZWwpKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBab25lLnByb3RvdHlwZS5jYW5jZWxUYXNrID0gZnVuY3Rpb24gKHRhc2spIHtcbiAgICAgICAgICAgICAgICBpZiAodGFzay56b25lICE9IHRoaXMpXG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignQSB0YXNrIGNhbiBvbmx5IGJlIGNhbmNlbGxlZCBpbiB0aGUgem9uZSBvZiBjcmVhdGlvbiEgKENyZWF0aW9uOiAnICtcbiAgICAgICAgICAgICAgICAgICAgICAgICh0YXNrLnpvbmUgfHwgTk9fWk9ORSkubmFtZSArICc7IEV4ZWN1dGlvbjogJyArIHRoaXMubmFtZSArICcpJyk7XG4gICAgICAgICAgICAgICAgaWYgKHRhc2suc3RhdGUgIT09IHNjaGVkdWxlZCAmJiB0YXNrLnN0YXRlICE9PSBydW5uaW5nKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdGFzay5fdHJhbnNpdGlvblRvKGNhbmNlbGluZywgc2NoZWR1bGVkLCBydW5uaW5nKTtcbiAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLl96b25lRGVsZWdhdGUuY2FuY2VsVGFzayh0aGlzLCB0YXNrKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgICAgICAgICAvLyBpZiBlcnJvciBvY2N1cnMgd2hlbiBjYW5jZWxUYXNrLCB0cmFuc2l0IHRoZSBzdGF0ZSB0byB1bmtub3duXG4gICAgICAgICAgICAgICAgICAgIHRhc2suX3RyYW5zaXRpb25Ubyh1bmtub3duLCBjYW5jZWxpbmcpO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLl96b25lRGVsZWdhdGUuaGFuZGxlRXJyb3IodGhpcywgZXJyKTtcbiAgICAgICAgICAgICAgICAgICAgdGhyb3cgZXJyO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB0aGlzLl91cGRhdGVUYXNrQ291bnQodGFzaywgLTEpO1xuICAgICAgICAgICAgICAgIHRhc2suX3RyYW5zaXRpb25Ubyhub3RTY2hlZHVsZWQsIGNhbmNlbGluZyk7XG4gICAgICAgICAgICAgICAgdGFzay5ydW5Db3VudCA9IDA7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRhc2s7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgWm9uZS5wcm90b3R5cGUuX3VwZGF0ZVRhc2tDb3VudCA9IGZ1bmN0aW9uICh0YXNrLCBjb3VudCkge1xuICAgICAgICAgICAgICAgIHZhciB6b25lRGVsZWdhdGVzID0gdGFzay5fem9uZURlbGVnYXRlcztcbiAgICAgICAgICAgICAgICBpZiAoY291bnQgPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgICAgdGFzay5fem9uZURlbGVnYXRlcyA9IG51bGw7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgem9uZURlbGVnYXRlcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICAgICAgICB6b25lRGVsZWdhdGVzW2ldLl91cGRhdGVUYXNrQ291bnQodGFzay50eXBlLCBjb3VudCk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHJldHVybiBab25lO1xuICAgICAgICB9KCkpO1xuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6cmVxdWlyZS1pbnRlcm5hbC13aXRoLXVuZGVyc2NvcmVcbiAgICAgICAgWm9uZS5fX3N5bWJvbF9fID0gX19zeW1ib2xfXztcbiAgICAgICAgdmFyIERFTEVHQVRFX1pTID0ge1xuICAgICAgICAgICAgbmFtZTogJycsXG4gICAgICAgICAgICBvbkhhc1Rhc2s6IGZ1bmN0aW9uIChkZWxlZ2F0ZSwgXywgdGFyZ2V0LCBoYXNUYXNrU3RhdGUpIHsgcmV0dXJuIGRlbGVnYXRlLmhhc1Rhc2sodGFyZ2V0LCBoYXNUYXNrU3RhdGUpOyB9LFxuICAgICAgICAgICAgb25TY2hlZHVsZVRhc2s6IGZ1bmN0aW9uIChkZWxlZ2F0ZSwgXywgdGFyZ2V0LCB0YXNrKSB7IHJldHVybiBkZWxlZ2F0ZS5zY2hlZHVsZVRhc2sodGFyZ2V0LCB0YXNrKTsgfSxcbiAgICAgICAgICAgIG9uSW52b2tlVGFzazogZnVuY3Rpb24gKGRlbGVnYXRlLCBfLCB0YXJnZXQsIHRhc2ssIGFwcGx5VGhpcywgYXBwbHlBcmdzKSB7IHJldHVybiBkZWxlZ2F0ZS5pbnZva2VUYXNrKHRhcmdldCwgdGFzaywgYXBwbHlUaGlzLCBhcHBseUFyZ3MpOyB9LFxuICAgICAgICAgICAgb25DYW5jZWxUYXNrOiBmdW5jdGlvbiAoZGVsZWdhdGUsIF8sIHRhcmdldCwgdGFzaykgeyByZXR1cm4gZGVsZWdhdGUuY2FuY2VsVGFzayh0YXJnZXQsIHRhc2spOyB9XG4gICAgICAgIH07XG4gICAgICAgIHZhciBfWm9uZURlbGVnYXRlID0gLyoqIEBjbGFzcyAqLyAoZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgZnVuY3Rpb24gX1pvbmVEZWxlZ2F0ZSh6b25lLCBwYXJlbnREZWxlZ2F0ZSwgem9uZVNwZWMpIHtcbiAgICAgICAgICAgICAgICB0aGlzLl90YXNrQ291bnRzID0geyAnbWljcm9UYXNrJzogMCwgJ21hY3JvVGFzayc6IDAsICdldmVudFRhc2snOiAwIH07XG4gICAgICAgICAgICAgICAgdGhpcy56b25lID0gem9uZTtcbiAgICAgICAgICAgICAgICB0aGlzLl9wYXJlbnREZWxlZ2F0ZSA9IHBhcmVudERlbGVnYXRlO1xuICAgICAgICAgICAgICAgIHRoaXMuX2ZvcmtaUyA9IHpvbmVTcGVjICYmICh6b25lU3BlYyAmJiB6b25lU3BlYy5vbkZvcmsgPyB6b25lU3BlYyA6IHBhcmVudERlbGVnYXRlLl9mb3JrWlMpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2ZvcmtEbGd0ID0gem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uRm9yayA/IHBhcmVudERlbGVnYXRlIDogcGFyZW50RGVsZWdhdGUuX2ZvcmtEbGd0KTtcbiAgICAgICAgICAgICAgICB0aGlzLl9mb3JrQ3VyclpvbmUgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25Gb3JrID8gdGhpcy56b25lIDogcGFyZW50RGVsZWdhdGUuX2ZvcmtDdXJyWm9uZSk7XG4gICAgICAgICAgICAgICAgdGhpcy5faW50ZXJjZXB0WlMgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25JbnRlcmNlcHQgPyB6b25lU3BlYyA6IHBhcmVudERlbGVnYXRlLl9pbnRlcmNlcHRaUyk7XG4gICAgICAgICAgICAgICAgdGhpcy5faW50ZXJjZXB0RGxndCA9XG4gICAgICAgICAgICAgICAgICAgIHpvbmVTcGVjICYmICh6b25lU3BlYy5vbkludGVyY2VwdCA/IHBhcmVudERlbGVnYXRlIDogcGFyZW50RGVsZWdhdGUuX2ludGVyY2VwdERsZ3QpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2ludGVyY2VwdEN1cnJab25lID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uSW50ZXJjZXB0ID8gdGhpcy56b25lIDogcGFyZW50RGVsZWdhdGUuX2ludGVyY2VwdEN1cnJab25lKTtcbiAgICAgICAgICAgICAgICB0aGlzLl9pbnZva2VaUyA9IHpvbmVTcGVjICYmICh6b25lU3BlYy5vbkludm9rZSA/IHpvbmVTcGVjIDogcGFyZW50RGVsZWdhdGUuX2ludm9rZVpTKTtcbiAgICAgICAgICAgICAgICB0aGlzLl9pbnZva2VEbGd0ID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uSW52b2tlID8gcGFyZW50RGVsZWdhdGUgOiBwYXJlbnREZWxlZ2F0ZS5faW52b2tlRGxndCk7XG4gICAgICAgICAgICAgICAgdGhpcy5faW52b2tlQ3VyclpvbmUgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25JbnZva2UgPyB0aGlzLnpvbmUgOiBwYXJlbnREZWxlZ2F0ZS5faW52b2tlQ3VyclpvbmUpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2hhbmRsZUVycm9yWlMgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25IYW5kbGVFcnJvciA/IHpvbmVTcGVjIDogcGFyZW50RGVsZWdhdGUuX2hhbmRsZUVycm9yWlMpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2hhbmRsZUVycm9yRGxndCA9XG4gICAgICAgICAgICAgICAgICAgIHpvbmVTcGVjICYmICh6b25lU3BlYy5vbkhhbmRsZUVycm9yID8gcGFyZW50RGVsZWdhdGUgOiBwYXJlbnREZWxlZ2F0ZS5faGFuZGxlRXJyb3JEbGd0KTtcbiAgICAgICAgICAgICAgICB0aGlzLl9oYW5kbGVFcnJvckN1cnJab25lID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uSGFuZGxlRXJyb3IgPyB0aGlzLnpvbmUgOiBwYXJlbnREZWxlZ2F0ZS5faGFuZGxlRXJyb3JDdXJyWm9uZSk7XG4gICAgICAgICAgICAgICAgdGhpcy5fc2NoZWR1bGVUYXNrWlMgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25TY2hlZHVsZVRhc2sgPyB6b25lU3BlYyA6IHBhcmVudERlbGVnYXRlLl9zY2hlZHVsZVRhc2taUyk7XG4gICAgICAgICAgICAgICAgdGhpcy5fc2NoZWR1bGVUYXNrRGxndCA9IHpvbmVTcGVjICYmXG4gICAgICAgICAgICAgICAgICAgICh6b25lU3BlYy5vblNjaGVkdWxlVGFzayA/IHBhcmVudERlbGVnYXRlIDogcGFyZW50RGVsZWdhdGUuX3NjaGVkdWxlVGFza0RsZ3QpO1xuICAgICAgICAgICAgICAgIHRoaXMuX3NjaGVkdWxlVGFza0N1cnJab25lID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uU2NoZWR1bGVUYXNrID8gdGhpcy56b25lIDogcGFyZW50RGVsZWdhdGUuX3NjaGVkdWxlVGFza0N1cnJab25lKTtcbiAgICAgICAgICAgICAgICB0aGlzLl9pbnZva2VUYXNrWlMgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25JbnZva2VUYXNrID8gem9uZVNwZWMgOiBwYXJlbnREZWxlZ2F0ZS5faW52b2tlVGFza1pTKTtcbiAgICAgICAgICAgICAgICB0aGlzLl9pbnZva2VUYXNrRGxndCA9XG4gICAgICAgICAgICAgICAgICAgIHpvbmVTcGVjICYmICh6b25lU3BlYy5vbkludm9rZVRhc2sgPyBwYXJlbnREZWxlZ2F0ZSA6IHBhcmVudERlbGVnYXRlLl9pbnZva2VUYXNrRGxndCk7XG4gICAgICAgICAgICAgICAgdGhpcy5faW52b2tlVGFza0N1cnJab25lID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uSW52b2tlVGFzayA/IHRoaXMuem9uZSA6IHBhcmVudERlbGVnYXRlLl9pbnZva2VUYXNrQ3VyclpvbmUpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2NhbmNlbFRhc2taUyA9XG4gICAgICAgICAgICAgICAgICAgIHpvbmVTcGVjICYmICh6b25lU3BlYy5vbkNhbmNlbFRhc2sgPyB6b25lU3BlYyA6IHBhcmVudERlbGVnYXRlLl9jYW5jZWxUYXNrWlMpO1xuICAgICAgICAgICAgICAgIHRoaXMuX2NhbmNlbFRhc2tEbGd0ID1cbiAgICAgICAgICAgICAgICAgICAgem9uZVNwZWMgJiYgKHpvbmVTcGVjLm9uQ2FuY2VsVGFzayA/IHBhcmVudERlbGVnYXRlIDogcGFyZW50RGVsZWdhdGUuX2NhbmNlbFRhc2tEbGd0KTtcbiAgICAgICAgICAgICAgICB0aGlzLl9jYW5jZWxUYXNrQ3VyclpvbmUgPVxuICAgICAgICAgICAgICAgICAgICB6b25lU3BlYyAmJiAoem9uZVNwZWMub25DYW5jZWxUYXNrID8gdGhpcy56b25lIDogcGFyZW50RGVsZWdhdGUuX2NhbmNlbFRhc2tDdXJyWm9uZSk7XG4gICAgICAgICAgICAgICAgdGhpcy5faGFzVGFza1pTID0gbnVsbDtcbiAgICAgICAgICAgICAgICB0aGlzLl9oYXNUYXNrRGxndCA9IG51bGw7XG4gICAgICAgICAgICAgICAgdGhpcy5faGFzVGFza0RsZ3RPd25lciA9IG51bGw7XG4gICAgICAgICAgICAgICAgdGhpcy5faGFzVGFza0N1cnJab25lID0gbnVsbDtcbiAgICAgICAgICAgICAgICB2YXIgem9uZVNwZWNIYXNUYXNrID0gem9uZVNwZWMgJiYgem9uZVNwZWMub25IYXNUYXNrO1xuICAgICAgICAgICAgICAgIHZhciBwYXJlbnRIYXNUYXNrID0gcGFyZW50RGVsZWdhdGUgJiYgcGFyZW50RGVsZWdhdGUuX2hhc1Rhc2taUztcbiAgICAgICAgICAgICAgICBpZiAoem9uZVNwZWNIYXNUYXNrIHx8IHBhcmVudEhhc1Rhc2spIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gSWYgd2UgbmVlZCB0byByZXBvcnQgaGFzVGFzaywgdGhhbiB0aGlzIFpTIG5lZWRzIHRvIGRvIHJlZiBjb3VudGluZyBvbiB0YXNrcy4gSW4gc3VjaFxuICAgICAgICAgICAgICAgICAgICAvLyBhIGNhc2UgYWxsIHRhc2sgcmVsYXRlZCBpbnRlcmNlcHRvcnMgbXVzdCBnbyB0aHJvdWdoIHRoaXMgWkQuIFdlIGNhbid0IHNob3J0IGNpcmN1aXQgaXQuXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2hhc1Rhc2taUyA9IHpvbmVTcGVjSGFzVGFzayA/IHpvbmVTcGVjIDogREVMRUdBVEVfWlM7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2hhc1Rhc2tEbGd0ID0gcGFyZW50RGVsZWdhdGU7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2hhc1Rhc2tEbGd0T3duZXIgPSB0aGlzO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLl9oYXNUYXNrQ3VyclpvbmUgPSB6b25lO1xuICAgICAgICAgICAgICAgICAgICBpZiAoIXpvbmVTcGVjLm9uU2NoZWR1bGVUYXNrKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl9zY2hlZHVsZVRhc2taUyA9IERFTEVHQVRFX1pTO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fc2NoZWR1bGVUYXNrRGxndCA9IHBhcmVudERlbGVnYXRlO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fc2NoZWR1bGVUYXNrQ3VyclpvbmUgPSB0aGlzLnpvbmU7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgaWYgKCF6b25lU3BlYy5vbkludm9rZVRhc2spIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuX2ludm9rZVRhc2taUyA9IERFTEVHQVRFX1pTO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5faW52b2tlVGFza0RsZ3QgPSBwYXJlbnREZWxlZ2F0ZTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuX2ludm9rZVRhc2tDdXJyWm9uZSA9IHRoaXMuem9uZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAoIXpvbmVTcGVjLm9uQ2FuY2VsVGFzaykge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fY2FuY2VsVGFza1pTID0gREVMRUdBVEVfWlM7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl9jYW5jZWxUYXNrRGxndCA9IHBhcmVudERlbGVnYXRlO1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fY2FuY2VsVGFza0N1cnJab25lID0gdGhpcy56b25lO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgX1pvbmVEZWxlZ2F0ZS5wcm90b3R5cGUuZm9yayA9IGZ1bmN0aW9uICh0YXJnZXRab25lLCB6b25lU3BlYykge1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9mb3JrWlMgPyB0aGlzLl9mb3JrWlMub25Gb3JrKHRoaXMuX2ZvcmtEbGd0LCB0aGlzLnpvbmUsIHRhcmdldFpvbmUsIHpvbmVTcGVjKSA6XG4gICAgICAgICAgICAgICAgICAgIG5ldyBab25lKHRhcmdldFpvbmUsIHpvbmVTcGVjKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBfWm9uZURlbGVnYXRlLnByb3RvdHlwZS5pbnRlcmNlcHQgPSBmdW5jdGlvbiAodGFyZ2V0Wm9uZSwgY2FsbGJhY2ssIHNvdXJjZSkge1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9pbnRlcmNlcHRaUyA/XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2ludGVyY2VwdFpTLm9uSW50ZXJjZXB0KHRoaXMuX2ludGVyY2VwdERsZ3QsIHRoaXMuX2ludGVyY2VwdEN1cnJab25lLCB0YXJnZXRab25lLCBjYWxsYmFjaywgc291cmNlKSA6XG4gICAgICAgICAgICAgICAgICAgIGNhbGxiYWNrO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIF9ab25lRGVsZWdhdGUucHJvdG90eXBlLmludm9rZSA9IGZ1bmN0aW9uICh0YXJnZXRab25lLCBjYWxsYmFjaywgYXBwbHlUaGlzLCBhcHBseUFyZ3MsIHNvdXJjZSkge1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9pbnZva2VaUyA/IHRoaXMuX2ludm9rZVpTLm9uSW52b2tlKHRoaXMuX2ludm9rZURsZ3QsIHRoaXMuX2ludm9rZUN1cnJab25lLCB0YXJnZXRab25lLCBjYWxsYmFjaywgYXBwbHlUaGlzLCBhcHBseUFyZ3MsIHNvdXJjZSkgOlxuICAgICAgICAgICAgICAgICAgICBjYWxsYmFjay5hcHBseShhcHBseVRoaXMsIGFwcGx5QXJncyk7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgX1pvbmVEZWxlZ2F0ZS5wcm90b3R5cGUuaGFuZGxlRXJyb3IgPSBmdW5jdGlvbiAodGFyZ2V0Wm9uZSwgZXJyb3IpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gdGhpcy5faGFuZGxlRXJyb3JaUyA/XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2hhbmRsZUVycm9yWlMub25IYW5kbGVFcnJvcih0aGlzLl9oYW5kbGVFcnJvckRsZ3QsIHRoaXMuX2hhbmRsZUVycm9yQ3VyclpvbmUsIHRhcmdldFpvbmUsIGVycm9yKSA6XG4gICAgICAgICAgICAgICAgICAgIHRydWU7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgX1pvbmVEZWxlZ2F0ZS5wcm90b3R5cGUuc2NoZWR1bGVUYXNrID0gZnVuY3Rpb24gKHRhcmdldFpvbmUsIHRhc2spIHtcbiAgICAgICAgICAgICAgICB2YXIgcmV0dXJuVGFzayA9IHRhc2s7XG4gICAgICAgICAgICAgICAgaWYgKHRoaXMuX3NjaGVkdWxlVGFza1pTKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0aGlzLl9oYXNUYXNrWlMpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVyblRhc2suX3pvbmVEZWxlZ2F0ZXMucHVzaCh0aGlzLl9oYXNUYXNrRGxndE93bmVyKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAvLyBjbGFuZy1mb3JtYXQgb2ZmXG4gICAgICAgICAgICAgICAgICAgIHJldHVyblRhc2sgPSB0aGlzLl9zY2hlZHVsZVRhc2taUy5vblNjaGVkdWxlVGFzayh0aGlzLl9zY2hlZHVsZVRhc2tEbGd0LCB0aGlzLl9zY2hlZHVsZVRhc2tDdXJyWm9uZSwgdGFyZ2V0Wm9uZSwgdGFzayk7XG4gICAgICAgICAgICAgICAgICAgIC8vIGNsYW5nLWZvcm1hdCBvblxuICAgICAgICAgICAgICAgICAgICBpZiAoIXJldHVyblRhc2spXG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm5UYXNrID0gdGFzaztcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0YXNrLnNjaGVkdWxlRm4pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suc2NoZWR1bGVGbih0YXNrKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHNlIGlmICh0YXNrLnR5cGUgPT0gbWljcm9UYXNrKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzY2hlZHVsZU1pY3JvVGFzayh0YXNrKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignVGFzayBpcyBtaXNzaW5nIHNjaGVkdWxlRm4uJyk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIHJldHVyblRhc2s7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgX1pvbmVEZWxlZ2F0ZS5wcm90b3R5cGUuaW52b2tlVGFzayA9IGZ1bmN0aW9uICh0YXJnZXRab25lLCB0YXNrLCBhcHBseVRoaXMsIGFwcGx5QXJncykge1xuICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9pbnZva2VUYXNrWlMgPyB0aGlzLl9pbnZva2VUYXNrWlMub25JbnZva2VUYXNrKHRoaXMuX2ludm9rZVRhc2tEbGd0LCB0aGlzLl9pbnZva2VUYXNrQ3VyclpvbmUsIHRhcmdldFpvbmUsIHRhc2ssIGFwcGx5VGhpcywgYXBwbHlBcmdzKSA6XG4gICAgICAgICAgICAgICAgICAgIHRhc2suY2FsbGJhY2suYXBwbHkoYXBwbHlUaGlzLCBhcHBseUFyZ3MpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIF9ab25lRGVsZWdhdGUucHJvdG90eXBlLmNhbmNlbFRhc2sgPSBmdW5jdGlvbiAodGFyZ2V0Wm9uZSwgdGFzaykge1xuICAgICAgICAgICAgICAgIHZhciB2YWx1ZTtcbiAgICAgICAgICAgICAgICBpZiAodGhpcy5fY2FuY2VsVGFza1pTKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhbHVlID0gdGhpcy5fY2FuY2VsVGFza1pTLm9uQ2FuY2VsVGFzayh0aGlzLl9jYW5jZWxUYXNrRGxndCwgdGhpcy5fY2FuY2VsVGFza0N1cnJab25lLCB0YXJnZXRab25lLCB0YXNrKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICghdGFzay5jYW5jZWxGbikge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgRXJyb3IoJ1Rhc2sgaXMgbm90IGNhbmNlbGFibGUnKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9IHRhc2suY2FuY2VsRm4odGFzayk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBfWm9uZURlbGVnYXRlLnByb3RvdHlwZS5oYXNUYXNrID0gZnVuY3Rpb24gKHRhcmdldFpvbmUsIGlzRW1wdHkpIHtcbiAgICAgICAgICAgICAgICAvLyBoYXNUYXNrIHNob3VsZCBub3QgdGhyb3cgZXJyb3Igc28gb3RoZXIgWm9uZURlbGVnYXRlXG4gICAgICAgICAgICAgICAgLy8gY2FuIHN0aWxsIHRyaWdnZXIgaGFzVGFzayBjYWxsYmFja1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuX2hhc1Rhc2taUyAmJlxuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5faGFzVGFza1pTLm9uSGFzVGFzayh0aGlzLl9oYXNUYXNrRGxndCwgdGhpcy5faGFzVGFza0N1cnJab25lLCB0YXJnZXRab25lLCBpc0VtcHR5KTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmhhbmRsZUVycm9yKHRhcmdldFpvbmUsIGVycik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpyZXF1aXJlLWludGVybmFsLXdpdGgtdW5kZXJzY29yZVxuICAgICAgICAgICAgX1pvbmVEZWxlZ2F0ZS5wcm90b3R5cGUuX3VwZGF0ZVRhc2tDb3VudCA9IGZ1bmN0aW9uICh0eXBlLCBjb3VudCkge1xuICAgICAgICAgICAgICAgIHZhciBjb3VudHMgPSB0aGlzLl90YXNrQ291bnRzO1xuICAgICAgICAgICAgICAgIHZhciBwcmV2ID0gY291bnRzW3R5cGVdO1xuICAgICAgICAgICAgICAgIHZhciBuZXh0ID0gY291bnRzW3R5cGVdID0gcHJldiArIGNvdW50O1xuICAgICAgICAgICAgICAgIGlmIChuZXh0IDwgMCkge1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ01vcmUgdGFza3MgZXhlY3V0ZWQgdGhlbiB3ZXJlIHNjaGVkdWxlZC4nKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKHByZXYgPT0gMCB8fCBuZXh0ID09IDApIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIGlzRW1wdHkgPSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBtaWNyb1Rhc2s6IGNvdW50c1snbWljcm9UYXNrJ10gPiAwLFxuICAgICAgICAgICAgICAgICAgICAgICAgbWFjcm9UYXNrOiBjb3VudHNbJ21hY3JvVGFzayddID4gMCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGV2ZW50VGFzazogY291bnRzWydldmVudFRhc2snXSA+IDAsXG4gICAgICAgICAgICAgICAgICAgICAgICBjaGFuZ2U6IHR5cGVcbiAgICAgICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5oYXNUYXNrKHRoaXMuem9uZSwgaXNFbXB0eSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHJldHVybiBfWm9uZURlbGVnYXRlO1xuICAgICAgICB9KCkpO1xuICAgICAgICB2YXIgWm9uZVRhc2sgPSAvKiogQGNsYXNzICovIChmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICBmdW5jdGlvbiBab25lVGFzayh0eXBlLCBzb3VyY2UsIGNhbGxiYWNrLCBvcHRpb25zLCBzY2hlZHVsZUZuLCBjYW5jZWxGbikge1xuICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpyZXF1aXJlLWludGVybmFsLXdpdGgtdW5kZXJzY29yZVxuICAgICAgICAgICAgICAgIHRoaXMuX3pvbmUgPSBudWxsO1xuICAgICAgICAgICAgICAgIHRoaXMucnVuQ291bnQgPSAwO1xuICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpyZXF1aXJlLWludGVybmFsLXdpdGgtdW5kZXJzY29yZVxuICAgICAgICAgICAgICAgIHRoaXMuX3pvbmVEZWxlZ2F0ZXMgPSBudWxsO1xuICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpyZXF1aXJlLWludGVybmFsLXdpdGgtdW5kZXJzY29yZVxuICAgICAgICAgICAgICAgIHRoaXMuX3N0YXRlID0gJ25vdFNjaGVkdWxlZCc7XG4gICAgICAgICAgICAgICAgdGhpcy50eXBlID0gdHlwZTtcbiAgICAgICAgICAgICAgICB0aGlzLnNvdXJjZSA9IHNvdXJjZTtcbiAgICAgICAgICAgICAgICB0aGlzLmRhdGEgPSBvcHRpb25zO1xuICAgICAgICAgICAgICAgIHRoaXMuc2NoZWR1bGVGbiA9IHNjaGVkdWxlRm47XG4gICAgICAgICAgICAgICAgdGhpcy5jYW5jZWxGbiA9IGNhbmNlbEZuO1xuICAgICAgICAgICAgICAgIGlmICghY2FsbGJhY2spIHtcbiAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdjYWxsYmFjayBpcyBub3QgZGVmaW5lZCcpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB0aGlzLmNhbGxiYWNrID0gY2FsbGJhY2s7XG4gICAgICAgICAgICAgICAgdmFyIHNlbGYgPSB0aGlzO1xuICAgICAgICAgICAgICAgIC8vIFRPRE86IEBKaWFMaVBhc3Npb24gb3B0aW9ucyBzaG91bGQgaGF2ZSBpbnRlcmZhY2VcbiAgICAgICAgICAgICAgICBpZiAodHlwZSA9PT0gZXZlbnRUYXNrICYmIG9wdGlvbnMgJiYgb3B0aW9ucy51c2VHKSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuaW52b2tlID0gWm9uZVRhc2suaW52b2tlVGFzaztcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuaW52b2tlID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFpvbmVUYXNrLmludm9rZVRhc2suY2FsbChnbG9iYWwsIHNlbGYsIHRoaXMsIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgWm9uZVRhc2suaW52b2tlVGFzayA9IGZ1bmN0aW9uICh0YXNrLCB0YXJnZXQsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICBpZiAoIXRhc2spIHtcbiAgICAgICAgICAgICAgICAgICAgdGFzayA9IHRoaXM7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIF9udW1iZXJPZk5lc3RlZFRhc2tGcmFtZXMrKztcbiAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICB0YXNrLnJ1bkNvdW50Kys7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0YXNrLnpvbmUucnVuVGFzayh0YXNrLCB0YXJnZXQsIGFyZ3MpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBmaW5hbGx5IHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKF9udW1iZXJPZk5lc3RlZFRhc2tGcmFtZXMgPT0gMSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgZHJhaW5NaWNyb1Rhc2tRdWV1ZSgpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIF9udW1iZXJPZk5lc3RlZFRhc2tGcmFtZXMtLTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KFpvbmVUYXNrLnByb3RvdHlwZSwgXCJ6b25lXCIsIHtcbiAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMuX3pvbmU7XG4gICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICBlbnVtZXJhYmxlOiBmYWxzZSxcbiAgICAgICAgICAgICAgICBjb25maWd1cmFibGU6IHRydWVcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KFpvbmVUYXNrLnByb3RvdHlwZSwgXCJzdGF0ZVwiLCB7XG4gICAgICAgICAgICAgICAgZ2V0OiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLl9zdGF0ZTtcbiAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgIGVudW1lcmFibGU6IGZhbHNlLFxuICAgICAgICAgICAgICAgIGNvbmZpZ3VyYWJsZTogdHJ1ZVxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBab25lVGFzay5wcm90b3R5cGUuY2FuY2VsU2NoZWR1bGVSZXF1ZXN0ID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHRoaXMuX3RyYW5zaXRpb25Ubyhub3RTY2hlZHVsZWQsIHNjaGVkdWxpbmcpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpyZXF1aXJlLWludGVybmFsLXdpdGgtdW5kZXJzY29yZVxuICAgICAgICAgICAgWm9uZVRhc2sucHJvdG90eXBlLl90cmFuc2l0aW9uVG8gPSBmdW5jdGlvbiAodG9TdGF0ZSwgZnJvbVN0YXRlMSwgZnJvbVN0YXRlMikge1xuICAgICAgICAgICAgICAgIGlmICh0aGlzLl9zdGF0ZSA9PT0gZnJvbVN0YXRlMSB8fCB0aGlzLl9zdGF0ZSA9PT0gZnJvbVN0YXRlMikge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLl9zdGF0ZSA9IHRvU3RhdGU7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0b1N0YXRlID09IG5vdFNjaGVkdWxlZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fem9uZURlbGVnYXRlcyA9IG51bGw7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcIlwiLmNvbmNhdCh0aGlzLnR5cGUsIFwiICdcIikuY29uY2F0KHRoaXMuc291cmNlLCBcIic6IGNhbiBub3QgdHJhbnNpdGlvbiB0byAnXCIpLmNvbmNhdCh0b1N0YXRlLCBcIicsIGV4cGVjdGluZyBzdGF0ZSAnXCIpLmNvbmNhdChmcm9tU3RhdGUxLCBcIidcIikuY29uY2F0KGZyb21TdGF0ZTIgPyAnIG9yIFxcJycgKyBmcm9tU3RhdGUyICsgJ1xcJycgOiAnJywgXCIsIHdhcyAnXCIpLmNvbmNhdCh0aGlzLl9zdGF0ZSwgXCInLlwiKSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVUYXNrLnByb3RvdHlwZS50b1N0cmluZyA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICBpZiAodGhpcy5kYXRhICYmIHR5cGVvZiB0aGlzLmRhdGEuaGFuZGxlSWQgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzLmRhdGEuaGFuZGxlSWQudG9TdHJpbmcoKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nLmNhbGwodGhpcyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIC8vIGFkZCB0b0pTT04gbWV0aG9kIHRvIHByZXZlbnQgY3ljbGljIGVycm9yIHdoZW5cbiAgICAgICAgICAgIC8vIGNhbGwgSlNPTi5zdHJpbmdpZnkoem9uZVRhc2spXG4gICAgICAgICAgICBab25lVGFzay5wcm90b3R5cGUudG9KU09OID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICAgICAgICAgIHR5cGU6IHRoaXMudHlwZSxcbiAgICAgICAgICAgICAgICAgICAgc3RhdGU6IHRoaXMuc3RhdGUsXG4gICAgICAgICAgICAgICAgICAgIHNvdXJjZTogdGhpcy5zb3VyY2UsXG4gICAgICAgICAgICAgICAgICAgIHpvbmU6IHRoaXMuem9uZS5uYW1lLFxuICAgICAgICAgICAgICAgICAgICBydW5Db3VudDogdGhpcy5ydW5Db3VudFxuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgcmV0dXJuIFpvbmVUYXNrO1xuICAgICAgICB9KCkpO1xuICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgICAgICAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4gICAgICAgIC8vLyAgTUlDUk9UQVNLIFFVRVVFXG4gICAgICAgIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgICAgICAgdmFyIHN5bWJvbFNldFRpbWVvdXQgPSBfX3N5bWJvbF9fKCdzZXRUaW1lb3V0Jyk7XG4gICAgICAgIHZhciBzeW1ib2xQcm9taXNlID0gX19zeW1ib2xfXygnUHJvbWlzZScpO1xuICAgICAgICB2YXIgc3ltYm9sVGhlbiA9IF9fc3ltYm9sX18oJ3RoZW4nKTtcbiAgICAgICAgdmFyIF9taWNyb1Rhc2tRdWV1ZSA9IFtdO1xuICAgICAgICB2YXIgX2lzRHJhaW5pbmdNaWNyb3Rhc2tRdWV1ZSA9IGZhbHNlO1xuICAgICAgICB2YXIgbmF0aXZlTWljcm9UYXNrUXVldWVQcm9taXNlO1xuICAgICAgICBmdW5jdGlvbiBuYXRpdmVTY2hlZHVsZU1pY3JvVGFzayhmdW5jKSB7XG4gICAgICAgICAgICBpZiAoIW5hdGl2ZU1pY3JvVGFza1F1ZXVlUHJvbWlzZSkge1xuICAgICAgICAgICAgICAgIGlmIChnbG9iYWxbc3ltYm9sUHJvbWlzZV0pIHtcbiAgICAgICAgICAgICAgICAgICAgbmF0aXZlTWljcm9UYXNrUXVldWVQcm9taXNlID0gZ2xvYmFsW3N5bWJvbFByb21pc2VdLnJlc29sdmUoMCk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKG5hdGl2ZU1pY3JvVGFza1F1ZXVlUHJvbWlzZSkge1xuICAgICAgICAgICAgICAgIHZhciBuYXRpdmVUaGVuID0gbmF0aXZlTWljcm9UYXNrUXVldWVQcm9taXNlW3N5bWJvbFRoZW5dO1xuICAgICAgICAgICAgICAgIGlmICghbmF0aXZlVGhlbikge1xuICAgICAgICAgICAgICAgICAgICAvLyBuYXRpdmUgUHJvbWlzZSBpcyBub3QgcGF0Y2hhYmxlLCB3ZSBuZWVkIHRvIHVzZSBgdGhlbmAgZGlyZWN0bHlcbiAgICAgICAgICAgICAgICAgICAgLy8gaXNzdWUgMTA3OFxuICAgICAgICAgICAgICAgICAgICBuYXRpdmVUaGVuID0gbmF0aXZlTWljcm9UYXNrUXVldWVQcm9taXNlWyd0aGVuJ107XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIG5hdGl2ZVRoZW4uY2FsbChuYXRpdmVNaWNyb1Rhc2tRdWV1ZVByb21pc2UsIGZ1bmMpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgZ2xvYmFsW3N5bWJvbFNldFRpbWVvdXRdKGZ1bmMsIDApO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGZ1bmN0aW9uIHNjaGVkdWxlTWljcm9UYXNrKHRhc2spIHtcbiAgICAgICAgICAgIC8vIGlmIHdlIGFyZSBub3QgcnVubmluZyBpbiBhbnkgdGFzaywgYW5kIHRoZXJlIGhhcyBub3QgYmVlbiBhbnl0aGluZyBzY2hlZHVsZWRcbiAgICAgICAgICAgIC8vIHdlIG11c3QgYm9vdHN0cmFwIHRoZSBpbml0aWFsIHRhc2sgY3JlYXRpb24gYnkgbWFudWFsbHkgc2NoZWR1bGluZyB0aGUgZHJhaW5cbiAgICAgICAgICAgIGlmIChfbnVtYmVyT2ZOZXN0ZWRUYXNrRnJhbWVzID09PSAwICYmIF9taWNyb1Rhc2tRdWV1ZS5sZW5ndGggPT09IDApIHtcbiAgICAgICAgICAgICAgICAvLyBXZSBhcmUgbm90IHJ1bm5pbmcgaW4gVGFzaywgc28gd2UgbmVlZCB0byBraWNrc3RhcnQgdGhlIG1pY3JvdGFzayBxdWV1ZS5cbiAgICAgICAgICAgICAgICBuYXRpdmVTY2hlZHVsZU1pY3JvVGFzayhkcmFpbk1pY3JvVGFza1F1ZXVlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHRhc2sgJiYgX21pY3JvVGFza1F1ZXVlLnB1c2godGFzayk7XG4gICAgICAgIH1cbiAgICAgICAgZnVuY3Rpb24gZHJhaW5NaWNyb1Rhc2tRdWV1ZSgpIHtcbiAgICAgICAgICAgIGlmICghX2lzRHJhaW5pbmdNaWNyb3Rhc2tRdWV1ZSkge1xuICAgICAgICAgICAgICAgIF9pc0RyYWluaW5nTWljcm90YXNrUXVldWUgPSB0cnVlO1xuICAgICAgICAgICAgICAgIHdoaWxlIChfbWljcm9UYXNrUXVldWUubGVuZ3RoKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciBxdWV1ZSA9IF9taWNyb1Rhc2tRdWV1ZTtcbiAgICAgICAgICAgICAgICAgICAgX21pY3JvVGFza1F1ZXVlID0gW107XG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcXVldWUubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0YXNrID0gcXVldWVbaV07XG4gICAgICAgICAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suem9uZS5ydW5UYXNrKHRhc2ssIG51bGwsIG51bGwpO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgX2FwaS5vblVuaGFuZGxlZEVycm9yKGVycm9yKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBfYXBpLm1pY3JvdGFza0RyYWluRG9uZSgpO1xuICAgICAgICAgICAgICAgIF9pc0RyYWluaW5nTWljcm90YXNrUXVldWUgPSBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgICAgICAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4gICAgICAgIC8vLyAgQk9PVFNUUkFQXG4gICAgICAgIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgICAgICAgdmFyIE5PX1pPTkUgPSB7IG5hbWU6ICdOTyBaT05FJyB9O1xuICAgICAgICB2YXIgbm90U2NoZWR1bGVkID0gJ25vdFNjaGVkdWxlZCcsIHNjaGVkdWxpbmcgPSAnc2NoZWR1bGluZycsIHNjaGVkdWxlZCA9ICdzY2hlZHVsZWQnLCBydW5uaW5nID0gJ3J1bm5pbmcnLCBjYW5jZWxpbmcgPSAnY2FuY2VsaW5nJywgdW5rbm93biA9ICd1bmtub3duJztcbiAgICAgICAgdmFyIG1pY3JvVGFzayA9ICdtaWNyb1Rhc2snLCBtYWNyb1Rhc2sgPSAnbWFjcm9UYXNrJywgZXZlbnRUYXNrID0gJ2V2ZW50VGFzayc7XG4gICAgICAgIHZhciBwYXRjaGVzID0ge307XG4gICAgICAgIHZhciBfYXBpID0ge1xuICAgICAgICAgICAgc3ltYm9sOiBfX3N5bWJvbF9fLFxuICAgICAgICAgICAgY3VycmVudFpvbmVGcmFtZTogZnVuY3Rpb24gKCkgeyByZXR1cm4gX2N1cnJlbnRab25lRnJhbWU7IH0sXG4gICAgICAgICAgICBvblVuaGFuZGxlZEVycm9yOiBub29wLFxuICAgICAgICAgICAgbWljcm90YXNrRHJhaW5Eb25lOiBub29wLFxuICAgICAgICAgICAgc2NoZWR1bGVNaWNyb1Rhc2s6IHNjaGVkdWxlTWljcm9UYXNrLFxuICAgICAgICAgICAgc2hvd1VuY2F1Z2h0RXJyb3I6IGZ1bmN0aW9uICgpIHsgcmV0dXJuICFab25lW19fc3ltYm9sX18oJ2lnbm9yZUNvbnNvbGVFcnJvclVuY2F1Z2h0RXJyb3InKV07IH0sXG4gICAgICAgICAgICBwYXRjaEV2ZW50VGFyZ2V0OiBmdW5jdGlvbiAoKSB7IHJldHVybiBbXTsgfSxcbiAgICAgICAgICAgIHBhdGNoT25Qcm9wZXJ0aWVzOiBub29wLFxuICAgICAgICAgICAgcGF0Y2hNZXRob2Q6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIG5vb3A7IH0sXG4gICAgICAgICAgICBiaW5kQXJndW1lbnRzOiBmdW5jdGlvbiAoKSB7IHJldHVybiBbXTsgfSxcbiAgICAgICAgICAgIHBhdGNoVGhlbjogZnVuY3Rpb24gKCkgeyByZXR1cm4gbm9vcDsgfSxcbiAgICAgICAgICAgIHBhdGNoTWFjcm9UYXNrOiBmdW5jdGlvbiAoKSB7IHJldHVybiBub29wOyB9LFxuICAgICAgICAgICAgcGF0Y2hFdmVudFByb3RvdHlwZTogZnVuY3Rpb24gKCkgeyByZXR1cm4gbm9vcDsgfSxcbiAgICAgICAgICAgIGlzSUVPckVkZ2U6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIGZhbHNlOyB9LFxuICAgICAgICAgICAgZ2V0R2xvYmFsT2JqZWN0czogZnVuY3Rpb24gKCkgeyByZXR1cm4gdW5kZWZpbmVkOyB9LFxuICAgICAgICAgICAgT2JqZWN0RGVmaW5lUHJvcGVydHk6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIG5vb3A7IH0sXG4gICAgICAgICAgICBPYmplY3RHZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3I6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHVuZGVmaW5lZDsgfSxcbiAgICAgICAgICAgIE9iamVjdENyZWF0ZTogZnVuY3Rpb24gKCkgeyByZXR1cm4gdW5kZWZpbmVkOyB9LFxuICAgICAgICAgICAgQXJyYXlTbGljZTogZnVuY3Rpb24gKCkgeyByZXR1cm4gW107IH0sXG4gICAgICAgICAgICBwYXRjaENsYXNzOiBmdW5jdGlvbiAoKSB7IHJldHVybiBub29wOyB9LFxuICAgICAgICAgICAgd3JhcFdpdGhDdXJyZW50Wm9uZTogZnVuY3Rpb24gKCkgeyByZXR1cm4gbm9vcDsgfSxcbiAgICAgICAgICAgIGZpbHRlclByb3BlcnRpZXM6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIFtdOyB9LFxuICAgICAgICAgICAgYXR0YWNoT3JpZ2luVG9QYXRjaGVkOiBmdW5jdGlvbiAoKSB7IHJldHVybiBub29wOyB9LFxuICAgICAgICAgICAgX3JlZGVmaW5lUHJvcGVydHk6IGZ1bmN0aW9uICgpIHsgcmV0dXJuIG5vb3A7IH0sXG4gICAgICAgICAgICBwYXRjaENhbGxiYWNrczogZnVuY3Rpb24gKCkgeyByZXR1cm4gbm9vcDsgfSxcbiAgICAgICAgICAgIG5hdGl2ZVNjaGVkdWxlTWljcm9UYXNrOiBuYXRpdmVTY2hlZHVsZU1pY3JvVGFza1xuICAgICAgICB9O1xuICAgICAgICB2YXIgX2N1cnJlbnRab25lRnJhbWUgPSB7IHBhcmVudDogbnVsbCwgem9uZTogbmV3IFpvbmUobnVsbCwgbnVsbCkgfTtcbiAgICAgICAgdmFyIF9jdXJyZW50VGFzayA9IG51bGw7XG4gICAgICAgIHZhciBfbnVtYmVyT2ZOZXN0ZWRUYXNrRnJhbWVzID0gMDtcbiAgICAgICAgZnVuY3Rpb24gbm9vcCgpIHsgfVxuICAgICAgICBwZXJmb3JtYW5jZU1lYXN1cmUoJ1pvbmUnLCAnWm9uZScpO1xuICAgICAgICByZXR1cm4gZ2xvYmFsWydab25lJ10gPSBab25lO1xuICAgIH0pKSh0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJyAmJiB3aW5kb3cgfHwgdHlwZW9mIHNlbGYgIT09ICd1bmRlZmluZWQnICYmIHNlbGYgfHwgZ2xvYmFsKTtcbiAgICAvKipcbiAgICAgKiBAbGljZW5zZVxuICAgICAqIENvcHlyaWdodCBHb29nbGUgTExDIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gICAgICpcbiAgICAgKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICAgICAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAgICAgKi9cbiAgICAvKipcbiAgICAgKiBTdXBwcmVzcyBjbG9zdXJlIGNvbXBpbGVyIGVycm9ycyBhYm91dCB1bmtub3duICdab25lJyB2YXJpYWJsZVxuICAgICAqIEBmaWxlb3ZlcnZpZXdcbiAgICAgKiBAc3VwcHJlc3Mge3VuZGVmaW5lZFZhcnMsZ2xvYmFsVGhpcyxtaXNzaW5nUmVxdWlyZX1cbiAgICAgKi9cbiAgICAvLy8gPHJlZmVyZW5jZSB0eXBlcz1cIm5vZGVcIi8+XG4gICAgLy8gaXNzdWUgIzk4OSwgdG8gcmVkdWNlIGJ1bmRsZSBzaXplLCB1c2Ugc2hvcnQgbmFtZVxuICAgIC8qKiBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yICovXG4gICAgdmFyIE9iamVjdEdldE93blByb3BlcnR5RGVzY3JpcHRvciA9IE9iamVjdC5nZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3I7XG4gICAgLyoqIE9iamVjdC5kZWZpbmVQcm9wZXJ0eSAqL1xuICAgIHZhciBPYmplY3REZWZpbmVQcm9wZXJ0eSA9IE9iamVjdC5kZWZpbmVQcm9wZXJ0eTtcbiAgICAvKiogT2JqZWN0LmdldFByb3RvdHlwZU9mICovXG4gICAgdmFyIE9iamVjdEdldFByb3RvdHlwZU9mID0gT2JqZWN0LmdldFByb3RvdHlwZU9mO1xuICAgIC8qKiBPYmplY3QuY3JlYXRlICovXG4gICAgdmFyIE9iamVjdENyZWF0ZSA9IE9iamVjdC5jcmVhdGU7XG4gICAgLyoqIEFycmF5LnByb3RvdHlwZS5zbGljZSAqL1xuICAgIHZhciBBcnJheVNsaWNlID0gQXJyYXkucHJvdG90eXBlLnNsaWNlO1xuICAgIC8qKiBhZGRFdmVudExpc3RlbmVyIHN0cmluZyBjb25zdCAqL1xuICAgIHZhciBBRERfRVZFTlRfTElTVEVORVJfU1RSID0gJ2FkZEV2ZW50TGlzdGVuZXInO1xuICAgIC8qKiByZW1vdmVFdmVudExpc3RlbmVyIHN0cmluZyBjb25zdCAqL1xuICAgIHZhciBSRU1PVkVfRVZFTlRfTElTVEVORVJfU1RSID0gJ3JlbW92ZUV2ZW50TGlzdGVuZXInO1xuICAgIC8qKiB6b25lU3ltYm9sIGFkZEV2ZW50TGlzdGVuZXIgKi9cbiAgICB2YXIgWk9ORV9TWU1CT0xfQUREX0VWRU5UX0xJU1RFTkVSID0gWm9uZS5fX3N5bWJvbF9fKEFERF9FVkVOVF9MSVNURU5FUl9TVFIpO1xuICAgIC8qKiB6b25lU3ltYm9sIHJlbW92ZUV2ZW50TGlzdGVuZXIgKi9cbiAgICB2YXIgWk9ORV9TWU1CT0xfUkVNT1ZFX0VWRU5UX0xJU1RFTkVSID0gWm9uZS5fX3N5bWJvbF9fKFJFTU9WRV9FVkVOVF9MSVNURU5FUl9TVFIpO1xuICAgIC8qKiB0cnVlIHN0cmluZyBjb25zdCAqL1xuICAgIHZhciBUUlVFX1NUUiA9ICd0cnVlJztcbiAgICAvKiogZmFsc2Ugc3RyaW5nIGNvbnN0ICovXG4gICAgdmFyIEZBTFNFX1NUUiA9ICdmYWxzZSc7XG4gICAgLyoqIFpvbmUgc3ltYm9sIHByZWZpeCBzdHJpbmcgY29uc3QuICovXG4gICAgdmFyIFpPTkVfU1lNQk9MX1BSRUZJWCA9IFpvbmUuX19zeW1ib2xfXygnJyk7XG4gICAgZnVuY3Rpb24gd3JhcFdpdGhDdXJyZW50Wm9uZShjYWxsYmFjaywgc291cmNlKSB7XG4gICAgICAgIHJldHVybiBab25lLmN1cnJlbnQud3JhcChjYWxsYmFjaywgc291cmNlKTtcbiAgICB9XG4gICAgZnVuY3Rpb24gc2NoZWR1bGVNYWNyb1Rhc2tXaXRoQ3VycmVudFpvbmUoc291cmNlLCBjYWxsYmFjaywgZGF0YSwgY3VzdG9tU2NoZWR1bGUsIGN1c3RvbUNhbmNlbCkge1xuICAgICAgICByZXR1cm4gWm9uZS5jdXJyZW50LnNjaGVkdWxlTWFjcm9UYXNrKHNvdXJjZSwgY2FsbGJhY2ssIGRhdGEsIGN1c3RvbVNjaGVkdWxlLCBjdXN0b21DYW5jZWwpO1xuICAgIH1cbiAgICB2YXIgem9uZVN5bWJvbCQxID0gWm9uZS5fX3N5bWJvbF9fO1xuICAgIHZhciBpc1dpbmRvd0V4aXN0cyA9IHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnO1xuICAgIHZhciBpbnRlcm5hbFdpbmRvdyA9IGlzV2luZG93RXhpc3RzID8gd2luZG93IDogdW5kZWZpbmVkO1xuICAgIHZhciBfZ2xvYmFsID0gaXNXaW5kb3dFeGlzdHMgJiYgaW50ZXJuYWxXaW5kb3cgfHwgdHlwZW9mIHNlbGYgPT09ICdvYmplY3QnICYmIHNlbGYgfHwgZ2xvYmFsO1xuICAgIHZhciBSRU1PVkVfQVRUUklCVVRFID0gJ3JlbW92ZUF0dHJpYnV0ZSc7XG4gICAgZnVuY3Rpb24gYmluZEFyZ3VtZW50cyhhcmdzLCBzb3VyY2UpIHtcbiAgICAgICAgZm9yICh2YXIgaSA9IGFyZ3MubGVuZ3RoIC0gMTsgaSA+PSAwOyBpLS0pIHtcbiAgICAgICAgICAgIGlmICh0eXBlb2YgYXJnc1tpXSA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgIGFyZ3NbaV0gPSB3cmFwV2l0aEN1cnJlbnRab25lKGFyZ3NbaV0sIHNvdXJjZSArICdfJyArIGkpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiBhcmdzO1xuICAgIH1cbiAgICBmdW5jdGlvbiBwYXRjaFByb3RvdHlwZShwcm90b3R5cGUsIGZuTmFtZXMpIHtcbiAgICAgICAgdmFyIHNvdXJjZSA9IHByb3RvdHlwZS5jb25zdHJ1Y3RvclsnbmFtZSddO1xuICAgICAgICB2YXIgX2xvb3BfMSA9IGZ1bmN0aW9uIChpKSB7XG4gICAgICAgICAgICB2YXIgbmFtZV8xID0gZm5OYW1lc1tpXTtcbiAgICAgICAgICAgIHZhciBkZWxlZ2F0ZSA9IHByb3RvdHlwZVtuYW1lXzFdO1xuICAgICAgICAgICAgaWYgKGRlbGVnYXRlKSB7XG4gICAgICAgICAgICAgICAgdmFyIHByb3RvdHlwZURlc2MgPSBPYmplY3RHZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3IocHJvdG90eXBlLCBuYW1lXzEpO1xuICAgICAgICAgICAgICAgIGlmICghaXNQcm9wZXJ0eVdyaXRhYmxlKHByb3RvdHlwZURlc2MpKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBcImNvbnRpbnVlXCI7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHByb3RvdHlwZVtuYW1lXzFdID0gKGZ1bmN0aW9uIChkZWxlZ2F0ZSkge1xuICAgICAgICAgICAgICAgICAgICB2YXIgcGF0Y2hlZCA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBkZWxlZ2F0ZS5hcHBseSh0aGlzLCBiaW5kQXJndW1lbnRzKGFyZ3VtZW50cywgc291cmNlICsgJy4nICsgbmFtZV8xKSk7XG4gICAgICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgICAgIGF0dGFjaE9yaWdpblRvUGF0Y2hlZChwYXRjaGVkLCBkZWxlZ2F0ZSk7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBwYXRjaGVkO1xuICAgICAgICAgICAgICAgIH0pKGRlbGVnYXRlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBmbk5hbWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICBfbG9vcF8xKGkpO1xuICAgICAgICB9XG4gICAgfVxuICAgIGZ1bmN0aW9uIGlzUHJvcGVydHlXcml0YWJsZShwcm9wZXJ0eURlc2MpIHtcbiAgICAgICAgaWYgKCFwcm9wZXJ0eURlc2MpIHtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIGlmIChwcm9wZXJ0eURlc2Mud3JpdGFibGUgPT09IGZhbHNlKSB7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuICEodHlwZW9mIHByb3BlcnR5RGVzYy5nZXQgPT09ICdmdW5jdGlvbicgJiYgdHlwZW9mIHByb3BlcnR5RGVzYy5zZXQgPT09ICd1bmRlZmluZWQnKTtcbiAgICB9XG4gICAgdmFyIGlzV2ViV29ya2VyID0gKHR5cGVvZiBXb3JrZXJHbG9iYWxTY29wZSAhPT0gJ3VuZGVmaW5lZCcgJiYgc2VsZiBpbnN0YW5jZW9mIFdvcmtlckdsb2JhbFNjb3BlKTtcbiAgICAvLyBNYWtlIHN1cmUgdG8gYWNjZXNzIGBwcm9jZXNzYCB0aHJvdWdoIGBfZ2xvYmFsYCBzbyB0aGF0IFdlYlBhY2sgZG9lcyBub3QgYWNjaWRlbnRhbGx5IGJyb3dzZXJpZnlcbiAgICAvLyB0aGlzIGNvZGUuXG4gICAgdmFyIGlzTm9kZSA9ICghKCdudycgaW4gX2dsb2JhbCkgJiYgdHlwZW9mIF9nbG9iYWwucHJvY2VzcyAhPT0gJ3VuZGVmaW5lZCcgJiZcbiAgICAgICAge30udG9TdHJpbmcuY2FsbChfZ2xvYmFsLnByb2Nlc3MpID09PSAnW29iamVjdCBwcm9jZXNzXScpO1xuICAgIHZhciBpc0Jyb3dzZXIgPSAhaXNOb2RlICYmICFpc1dlYldvcmtlciAmJiAhIShpc1dpbmRvd0V4aXN0cyAmJiBpbnRlcm5hbFdpbmRvd1snSFRNTEVsZW1lbnQnXSk7XG4gICAgLy8gd2UgYXJlIGluIGVsZWN0cm9uIG9mIG53LCBzbyB3ZSBhcmUgYm90aCBicm93c2VyIGFuZCBub2RlanNcbiAgICAvLyBNYWtlIHN1cmUgdG8gYWNjZXNzIGBwcm9jZXNzYCB0aHJvdWdoIGBfZ2xvYmFsYCBzbyB0aGF0IFdlYlBhY2sgZG9lcyBub3QgYWNjaWRlbnRhbGx5IGJyb3dzZXJpZnlcbiAgICAvLyB0aGlzIGNvZGUuXG4gICAgdmFyIGlzTWl4ID0gdHlwZW9mIF9nbG9iYWwucHJvY2VzcyAhPT0gJ3VuZGVmaW5lZCcgJiZcbiAgICAgICAge30udG9TdHJpbmcuY2FsbChfZ2xvYmFsLnByb2Nlc3MpID09PSAnW29iamVjdCBwcm9jZXNzXScgJiYgIWlzV2ViV29ya2VyICYmXG4gICAgICAgICEhKGlzV2luZG93RXhpc3RzICYmIGludGVybmFsV2luZG93WydIVE1MRWxlbWVudCddKTtcbiAgICB2YXIgem9uZVN5bWJvbEV2ZW50TmFtZXMkMSA9IHt9O1xuICAgIHZhciB3cmFwRm4gPSBmdW5jdGlvbiAoZXZlbnQpIHtcbiAgICAgICAgLy8gaHR0cHM6Ly9naXRodWIuY29tL2FuZ3VsYXIvem9uZS5qcy9pc3N1ZXMvOTExLCBpbiBJRSwgc29tZXRpbWVzXG4gICAgICAgIC8vIGV2ZW50IHdpbGwgYmUgdW5kZWZpbmVkLCBzbyB3ZSBuZWVkIHRvIHVzZSB3aW5kb3cuZXZlbnRcbiAgICAgICAgZXZlbnQgPSBldmVudCB8fCBfZ2xvYmFsLmV2ZW50O1xuICAgICAgICBpZiAoIWV2ZW50KSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdmFyIGV2ZW50TmFtZVN5bWJvbCA9IHpvbmVTeW1ib2xFdmVudE5hbWVzJDFbZXZlbnQudHlwZV07XG4gICAgICAgIGlmICghZXZlbnROYW1lU3ltYm9sKSB7XG4gICAgICAgICAgICBldmVudE5hbWVTeW1ib2wgPSB6b25lU3ltYm9sRXZlbnROYW1lcyQxW2V2ZW50LnR5cGVdID0gem9uZVN5bWJvbCQxKCdPTl9QUk9QRVJUWScgKyBldmVudC50eXBlKTtcbiAgICAgICAgfVxuICAgICAgICB2YXIgdGFyZ2V0ID0gdGhpcyB8fCBldmVudC50YXJnZXQgfHwgX2dsb2JhbDtcbiAgICAgICAgdmFyIGxpc3RlbmVyID0gdGFyZ2V0W2V2ZW50TmFtZVN5bWJvbF07XG4gICAgICAgIHZhciByZXN1bHQ7XG4gICAgICAgIGlmIChpc0Jyb3dzZXIgJiYgdGFyZ2V0ID09PSBpbnRlcm5hbFdpbmRvdyAmJiBldmVudC50eXBlID09PSAnZXJyb3InKSB7XG4gICAgICAgICAgICAvLyB3aW5kb3cub25lcnJvciBoYXZlIGRpZmZlcmVudCBzaWduYXR1cmVcbiAgICAgICAgICAgIC8vIGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0FQSS9HbG9iYWxFdmVudEhhbmRsZXJzL29uZXJyb3Ijd2luZG93Lm9uZXJyb3JcbiAgICAgICAgICAgIC8vIGFuZCBvbmVycm9yIGNhbGxiYWNrIHdpbGwgcHJldmVudCBkZWZhdWx0IHdoZW4gY2FsbGJhY2sgcmV0dXJuIHRydWVcbiAgICAgICAgICAgIHZhciBlcnJvckV2ZW50ID0gZXZlbnQ7XG4gICAgICAgICAgICByZXN1bHQgPSBsaXN0ZW5lciAmJlxuICAgICAgICAgICAgICAgIGxpc3RlbmVyLmNhbGwodGhpcywgZXJyb3JFdmVudC5tZXNzYWdlLCBlcnJvckV2ZW50LmZpbGVuYW1lLCBlcnJvckV2ZW50LmxpbmVubywgZXJyb3JFdmVudC5jb2xubywgZXJyb3JFdmVudC5lcnJvcik7XG4gICAgICAgICAgICBpZiAocmVzdWx0ID09PSB0cnVlKSB7XG4gICAgICAgICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIHJlc3VsdCA9IGxpc3RlbmVyICYmIGxpc3RlbmVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICBpZiAocmVzdWx0ICE9IHVuZGVmaW5lZCAmJiAhcmVzdWx0KSB7XG4gICAgICAgICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH07XG4gICAgZnVuY3Rpb24gcGF0Y2hQcm9wZXJ0eShvYmosIHByb3AsIHByb3RvdHlwZSkge1xuICAgICAgICB2YXIgZGVzYyA9IE9iamVjdEdldE93blByb3BlcnR5RGVzY3JpcHRvcihvYmosIHByb3ApO1xuICAgICAgICBpZiAoIWRlc2MgJiYgcHJvdG90eXBlKSB7XG4gICAgICAgICAgICAvLyB3aGVuIHBhdGNoIHdpbmRvdyBvYmplY3QsIHVzZSBwcm90b3R5cGUgdG8gY2hlY2sgcHJvcCBleGlzdCBvciBub3RcbiAgICAgICAgICAgIHZhciBwcm90b3R5cGVEZXNjID0gT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHByb3RvdHlwZSwgcHJvcCk7XG4gICAgICAgICAgICBpZiAocHJvdG90eXBlRGVzYykge1xuICAgICAgICAgICAgICAgIGRlc2MgPSB7IGVudW1lcmFibGU6IHRydWUsIGNvbmZpZ3VyYWJsZTogdHJ1ZSB9O1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIC8vIGlmIHRoZSBkZXNjcmlwdG9yIG5vdCBleGlzdHMgb3IgaXMgbm90IGNvbmZpZ3VyYWJsZVxuICAgICAgICAvLyBqdXN0IHJldHVyblxuICAgICAgICBpZiAoIWRlc2MgfHwgIWRlc2MuY29uZmlndXJhYmxlKSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdmFyIG9uUHJvcFBhdGNoZWRTeW1ib2wgPSB6b25lU3ltYm9sJDEoJ29uJyArIHByb3AgKyAncGF0Y2hlZCcpO1xuICAgICAgICBpZiAob2JqLmhhc093blByb3BlcnR5KG9uUHJvcFBhdGNoZWRTeW1ib2wpICYmIG9ialtvblByb3BQYXRjaGVkU3ltYm9sXSkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIC8vIEEgcHJvcGVydHkgZGVzY3JpcHRvciBjYW5ub3QgaGF2ZSBnZXR0ZXIvc2V0dGVyIGFuZCBiZSB3cml0YWJsZVxuICAgICAgICAvLyBkZWxldGluZyB0aGUgd3JpdGFibGUgYW5kIHZhbHVlIHByb3BlcnRpZXMgYXZvaWRzIHRoaXMgZXJyb3I6XG4gICAgICAgIC8vXG4gICAgICAgIC8vIFR5cGVFcnJvcjogcHJvcGVydHkgZGVzY3JpcHRvcnMgbXVzdCBub3Qgc3BlY2lmeSBhIHZhbHVlIG9yIGJlIHdyaXRhYmxlIHdoZW4gYVxuICAgICAgICAvLyBnZXR0ZXIgb3Igc2V0dGVyIGhhcyBiZWVuIHNwZWNpZmllZFxuICAgICAgICBkZWxldGUgZGVzYy53cml0YWJsZTtcbiAgICAgICAgZGVsZXRlIGRlc2MudmFsdWU7XG4gICAgICAgIHZhciBvcmlnaW5hbERlc2NHZXQgPSBkZXNjLmdldDtcbiAgICAgICAgdmFyIG9yaWdpbmFsRGVzY1NldCA9IGRlc2Muc2V0O1xuICAgICAgICAvLyBzbGljZSgyKSBjdXogJ29uY2xpY2snIC0+ICdjbGljaycsIGV0Y1xuICAgICAgICB2YXIgZXZlbnROYW1lID0gcHJvcC5zbGljZSgyKTtcbiAgICAgICAgdmFyIGV2ZW50TmFtZVN5bWJvbCA9IHpvbmVTeW1ib2xFdmVudE5hbWVzJDFbZXZlbnROYW1lXTtcbiAgICAgICAgaWYgKCFldmVudE5hbWVTeW1ib2wpIHtcbiAgICAgICAgICAgIGV2ZW50TmFtZVN5bWJvbCA9IHpvbmVTeW1ib2xFdmVudE5hbWVzJDFbZXZlbnROYW1lXSA9IHpvbmVTeW1ib2wkMSgnT05fUFJPUEVSVFknICsgZXZlbnROYW1lKTtcbiAgICAgICAgfVxuICAgICAgICBkZXNjLnNldCA9IGZ1bmN0aW9uIChuZXdWYWx1ZSkge1xuICAgICAgICAgICAgLy8gaW4gc29tZSBvZiB3aW5kb3dzJ3Mgb25wcm9wZXJ0eSBjYWxsYmFjaywgdGhpcyBpcyB1bmRlZmluZWRcbiAgICAgICAgICAgIC8vIHNvIHdlIG5lZWQgdG8gY2hlY2sgaXRcbiAgICAgICAgICAgIHZhciB0YXJnZXQgPSB0aGlzO1xuICAgICAgICAgICAgaWYgKCF0YXJnZXQgJiYgb2JqID09PSBfZ2xvYmFsKSB7XG4gICAgICAgICAgICAgICAgdGFyZ2V0ID0gX2dsb2JhbDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmICghdGFyZ2V0KSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIHByZXZpb3VzVmFsdWUgPSB0YXJnZXRbZXZlbnROYW1lU3ltYm9sXTtcbiAgICAgICAgICAgIGlmICh0eXBlb2YgcHJldmlvdXNWYWx1ZSA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgIHRhcmdldC5yZW1vdmVFdmVudExpc3RlbmVyKGV2ZW50TmFtZSwgd3JhcEZuKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8vIGlzc3VlICM5NzgsIHdoZW4gb25sb2FkIGhhbmRsZXIgd2FzIGFkZGVkIGJlZm9yZSBsb2FkaW5nIHpvbmUuanNcbiAgICAgICAgICAgIC8vIHdlIHNob3VsZCByZW1vdmUgaXQgd2l0aCBvcmlnaW5hbERlc2NTZXRcbiAgICAgICAgICAgIG9yaWdpbmFsRGVzY1NldCAmJiBvcmlnaW5hbERlc2NTZXQuY2FsbCh0YXJnZXQsIG51bGwpO1xuICAgICAgICAgICAgdGFyZ2V0W2V2ZW50TmFtZVN5bWJvbF0gPSBuZXdWYWx1ZTtcbiAgICAgICAgICAgIGlmICh0eXBlb2YgbmV3VmFsdWUgPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICB0YXJnZXQuYWRkRXZlbnRMaXN0ZW5lcihldmVudE5hbWUsIHdyYXBGbiwgZmFsc2UpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9O1xuICAgICAgICAvLyBUaGUgZ2V0dGVyIHdvdWxkIHJldHVybiB1bmRlZmluZWQgZm9yIHVuYXNzaWduZWQgcHJvcGVydGllcyBidXQgdGhlIGRlZmF1bHQgdmFsdWUgb2YgYW5cbiAgICAgICAgLy8gdW5hc3NpZ25lZCBwcm9wZXJ0eSBpcyBudWxsXG4gICAgICAgIGRlc2MuZ2V0ID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgLy8gaW4gc29tZSBvZiB3aW5kb3dzJ3Mgb25wcm9wZXJ0eSBjYWxsYmFjaywgdGhpcyBpcyB1bmRlZmluZWRcbiAgICAgICAgICAgIC8vIHNvIHdlIG5lZWQgdG8gY2hlY2sgaXRcbiAgICAgICAgICAgIHZhciB0YXJnZXQgPSB0aGlzO1xuICAgICAgICAgICAgaWYgKCF0YXJnZXQgJiYgb2JqID09PSBfZ2xvYmFsKSB7XG4gICAgICAgICAgICAgICAgdGFyZ2V0ID0gX2dsb2JhbDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmICghdGFyZ2V0KSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgbGlzdGVuZXIgPSB0YXJnZXRbZXZlbnROYW1lU3ltYm9sXTtcbiAgICAgICAgICAgIGlmIChsaXN0ZW5lcikge1xuICAgICAgICAgICAgICAgIHJldHVybiBsaXN0ZW5lcjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGVsc2UgaWYgKG9yaWdpbmFsRGVzY0dldCkge1xuICAgICAgICAgICAgICAgIC8vIHJlc3VsdCB3aWxsIGJlIG51bGwgd2hlbiB1c2UgaW5saW5lIGV2ZW50IGF0dHJpYnV0ZSxcbiAgICAgICAgICAgICAgICAvLyBzdWNoIGFzIDxidXR0b24gb25jbGljaz1cImZ1bmMoKTtcIj5PSzwvYnV0dG9uPlxuICAgICAgICAgICAgICAgIC8vIGJlY2F1c2UgdGhlIG9uY2xpY2sgZnVuY3Rpb24gaXMgaW50ZXJuYWwgcmF3IHVuY29tcGlsZWQgaGFuZGxlclxuICAgICAgICAgICAgICAgIC8vIHRoZSBvbmNsaWNrIHdpbGwgYmUgZXZhbHVhdGVkIHdoZW4gZmlyc3QgdGltZSBldmVudCB3YXMgdHJpZ2dlcmVkIG9yXG4gICAgICAgICAgICAgICAgLy8gdGhlIHByb3BlcnR5IGlzIGFjY2Vzc2VkLCBodHRwczovL2dpdGh1Yi5jb20vYW5ndWxhci96b25lLmpzL2lzc3Vlcy81MjVcbiAgICAgICAgICAgICAgICAvLyBzbyB3ZSBzaG91bGQgdXNlIG9yaWdpbmFsIG5hdGl2ZSBnZXQgdG8gcmV0cmlldmUgdGhlIGhhbmRsZXJcbiAgICAgICAgICAgICAgICB2YXIgdmFsdWUgPSBvcmlnaW5hbERlc2NHZXQuY2FsbCh0aGlzKTtcbiAgICAgICAgICAgICAgICBpZiAodmFsdWUpIHtcbiAgICAgICAgICAgICAgICAgICAgZGVzYy5zZXQuY2FsbCh0aGlzLCB2YWx1ZSk7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgdGFyZ2V0W1JFTU9WRV9BVFRSSUJVVEVdID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0YXJnZXQucmVtb3ZlQXR0cmlidXRlKHByb3ApO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfTtcbiAgICAgICAgT2JqZWN0RGVmaW5lUHJvcGVydHkob2JqLCBwcm9wLCBkZXNjKTtcbiAgICAgICAgb2JqW29uUHJvcFBhdGNoZWRTeW1ib2xdID0gdHJ1ZTtcbiAgICB9XG4gICAgZnVuY3Rpb24gcGF0Y2hPblByb3BlcnRpZXMob2JqLCBwcm9wZXJ0aWVzLCBwcm90b3R5cGUpIHtcbiAgICAgICAgaWYgKHByb3BlcnRpZXMpIHtcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcHJvcGVydGllcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICAgIHBhdGNoUHJvcGVydHkob2JqLCAnb24nICsgcHJvcGVydGllc1tpXSwgcHJvdG90eXBlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIHZhciBvblByb3BlcnRpZXMgPSBbXTtcbiAgICAgICAgICAgIGZvciAodmFyIHByb3AgaW4gb2JqKSB7XG4gICAgICAgICAgICAgICAgaWYgKHByb3Auc2xpY2UoMCwgMikgPT0gJ29uJykge1xuICAgICAgICAgICAgICAgICAgICBvblByb3BlcnRpZXMucHVzaChwcm9wKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IG9uUHJvcGVydGllcy5sZW5ndGg7IGorKykge1xuICAgICAgICAgICAgICAgIHBhdGNoUHJvcGVydHkob2JqLCBvblByb3BlcnRpZXNbal0sIHByb3RvdHlwZSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG4gICAgdmFyIG9yaWdpbmFsSW5zdGFuY2VLZXkgPSB6b25lU3ltYm9sJDEoJ29yaWdpbmFsSW5zdGFuY2UnKTtcbiAgICAvLyB3cmFwIHNvbWUgbmF0aXZlIEFQSSBvbiBgd2luZG93YFxuICAgIGZ1bmN0aW9uIHBhdGNoQ2xhc3MoY2xhc3NOYW1lKSB7XG4gICAgICAgIHZhciBPcmlnaW5hbENsYXNzID0gX2dsb2JhbFtjbGFzc05hbWVdO1xuICAgICAgICBpZiAoIU9yaWdpbmFsQ2xhc3MpXG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIC8vIGtlZXAgb3JpZ2luYWwgY2xhc3MgaW4gZ2xvYmFsXG4gICAgICAgIF9nbG9iYWxbem9uZVN5bWJvbCQxKGNsYXNzTmFtZSldID0gT3JpZ2luYWxDbGFzcztcbiAgICAgICAgX2dsb2JhbFtjbGFzc05hbWVdID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgdmFyIGEgPSBiaW5kQXJndW1lbnRzKGFyZ3VtZW50cywgY2xhc3NOYW1lKTtcbiAgICAgICAgICAgIHN3aXRjaCAoYS5sZW5ndGgpIHtcbiAgICAgICAgICAgICAgICBjYXNlIDA6XG4gICAgICAgICAgICAgICAgICAgIHRoaXNbb3JpZ2luYWxJbnN0YW5jZUtleV0gPSBuZXcgT3JpZ2luYWxDbGFzcygpO1xuICAgICAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgICBjYXNlIDE6XG4gICAgICAgICAgICAgICAgICAgIHRoaXNbb3JpZ2luYWxJbnN0YW5jZUtleV0gPSBuZXcgT3JpZ2luYWxDbGFzcyhhWzBdKTtcbiAgICAgICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgICAgY2FzZSAyOlxuICAgICAgICAgICAgICAgICAgICB0aGlzW29yaWdpbmFsSW5zdGFuY2VLZXldID0gbmV3IE9yaWdpbmFsQ2xhc3MoYVswXSwgYVsxXSk7XG4gICAgICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICAgIGNhc2UgMzpcbiAgICAgICAgICAgICAgICAgICAgdGhpc1tvcmlnaW5hbEluc3RhbmNlS2V5XSA9IG5ldyBPcmlnaW5hbENsYXNzKGFbMF0sIGFbMV0sIGFbMl0pO1xuICAgICAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgICBjYXNlIDQ6XG4gICAgICAgICAgICAgICAgICAgIHRoaXNbb3JpZ2luYWxJbnN0YW5jZUtleV0gPSBuZXcgT3JpZ2luYWxDbGFzcyhhWzBdLCBhWzFdLCBhWzJdLCBhWzNdKTtcbiAgICAgICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgICAgZGVmYXVsdDpcbiAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdBcmcgbGlzdCB0b28gbG9uZy4nKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTtcbiAgICAgICAgLy8gYXR0YWNoIG9yaWdpbmFsIGRlbGVnYXRlIHRvIHBhdGNoZWQgZnVuY3Rpb25cbiAgICAgICAgYXR0YWNoT3JpZ2luVG9QYXRjaGVkKF9nbG9iYWxbY2xhc3NOYW1lXSwgT3JpZ2luYWxDbGFzcyk7XG4gICAgICAgIHZhciBpbnN0YW5jZSA9IG5ldyBPcmlnaW5hbENsYXNzKGZ1bmN0aW9uICgpIHsgfSk7XG4gICAgICAgIHZhciBwcm9wO1xuICAgICAgICBmb3IgKHByb3AgaW4gaW5zdGFuY2UpIHtcbiAgICAgICAgICAgIC8vIGh0dHBzOi8vYnVncy53ZWJraXQub3JnL3Nob3dfYnVnLmNnaT9pZD00NDcyMVxuICAgICAgICAgICAgaWYgKGNsYXNzTmFtZSA9PT0gJ1hNTEh0dHBSZXF1ZXN0JyAmJiBwcm9wID09PSAncmVzcG9uc2VCbG9iJylcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIChmdW5jdGlvbiAocHJvcCkge1xuICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgaW5zdGFuY2VbcHJvcF0gPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICAgICAgX2dsb2JhbFtjbGFzc05hbWVdLnByb3RvdHlwZVtwcm9wXSA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzW29yaWdpbmFsSW5zdGFuY2VLZXldW3Byb3BdLmFwcGx5KHRoaXNbb3JpZ2luYWxJbnN0YW5jZUtleV0sIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBPYmplY3REZWZpbmVQcm9wZXJ0eShfZ2xvYmFsW2NsYXNzTmFtZV0ucHJvdG90eXBlLCBwcm9wLCB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzZXQ6IGZ1bmN0aW9uIChmbikge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgZm4gPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpc1tvcmlnaW5hbEluc3RhbmNlS2V5XVtwcm9wXSA9IHdyYXBXaXRoQ3VycmVudFpvbmUoZm4sIGNsYXNzTmFtZSArICcuJyArIHByb3ApO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBrZWVwIGNhbGxiYWNrIGluIHdyYXBwZWQgZnVuY3Rpb24gc28gd2UgY2FuXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIHVzZSBpdCBpbiBGdW5jdGlvbi5wcm90b3R5cGUudG9TdHJpbmcgdG8gcmV0dXJuXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIHRoZSBuYXRpdmUgb25lLlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhdHRhY2hPcmlnaW5Ub1BhdGNoZWQodGhpc1tvcmlnaW5hbEluc3RhbmNlS2V5XVtwcm9wXSwgZm4pO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpc1tvcmlnaW5hbEluc3RhbmNlS2V5XVtwcm9wXSA9IGZuO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdGhpc1tvcmlnaW5hbEluc3RhbmNlS2V5XVtwcm9wXTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfShwcm9wKSk7XG4gICAgICAgIH1cbiAgICAgICAgZm9yIChwcm9wIGluIE9yaWdpbmFsQ2xhc3MpIHtcbiAgICAgICAgICAgIGlmIChwcm9wICE9PSAncHJvdG90eXBlJyAmJiBPcmlnaW5hbENsYXNzLmhhc093blByb3BlcnR5KHByb3ApKSB7XG4gICAgICAgICAgICAgICAgX2dsb2JhbFtjbGFzc05hbWVdW3Byb3BdID0gT3JpZ2luYWxDbGFzc1twcm9wXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cbiAgICBmdW5jdGlvbiBwYXRjaE1ldGhvZCh0YXJnZXQsIG5hbWUsIHBhdGNoRm4pIHtcbiAgICAgICAgdmFyIHByb3RvID0gdGFyZ2V0O1xuICAgICAgICB3aGlsZSAocHJvdG8gJiYgIXByb3RvLmhhc093blByb3BlcnR5KG5hbWUpKSB7XG4gICAgICAgICAgICBwcm90byA9IE9iamVjdEdldFByb3RvdHlwZU9mKHByb3RvKTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoIXByb3RvICYmIHRhcmdldFtuYW1lXSkge1xuICAgICAgICAgICAgLy8gc29tZWhvdyB3ZSBkaWQgbm90IGZpbmQgaXQsIGJ1dCB3ZSBjYW4gc2VlIGl0LiBUaGlzIGhhcHBlbnMgb24gSUUgZm9yIFdpbmRvdyBwcm9wZXJ0aWVzLlxuICAgICAgICAgICAgcHJvdG8gPSB0YXJnZXQ7XG4gICAgICAgIH1cbiAgICAgICAgdmFyIGRlbGVnYXRlTmFtZSA9IHpvbmVTeW1ib2wkMShuYW1lKTtcbiAgICAgICAgdmFyIGRlbGVnYXRlID0gbnVsbDtcbiAgICAgICAgaWYgKHByb3RvICYmICghKGRlbGVnYXRlID0gcHJvdG9bZGVsZWdhdGVOYW1lXSkgfHwgIXByb3RvLmhhc093blByb3BlcnR5KGRlbGVnYXRlTmFtZSkpKSB7XG4gICAgICAgICAgICBkZWxlZ2F0ZSA9IHByb3RvW2RlbGVnYXRlTmFtZV0gPSBwcm90b1tuYW1lXTtcbiAgICAgICAgICAgIC8vIGNoZWNrIHdoZXRoZXIgcHJvdG9bbmFtZV0gaXMgd3JpdGFibGVcbiAgICAgICAgICAgIC8vIHNvbWUgcHJvcGVydHkgaXMgcmVhZG9ubHkgaW4gc2FmYXJpLCBzdWNoIGFzIEh0bWxDYW52YXNFbGVtZW50LnByb3RvdHlwZS50b0Jsb2JcbiAgICAgICAgICAgIHZhciBkZXNjID0gcHJvdG8gJiYgT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHByb3RvLCBuYW1lKTtcbiAgICAgICAgICAgIGlmIChpc1Byb3BlcnR5V3JpdGFibGUoZGVzYykpIHtcbiAgICAgICAgICAgICAgICB2YXIgcGF0Y2hEZWxlZ2F0ZV8xID0gcGF0Y2hGbihkZWxlZ2F0ZSwgZGVsZWdhdGVOYW1lLCBuYW1lKTtcbiAgICAgICAgICAgICAgICBwcm90b1tuYW1lXSA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHBhdGNoRGVsZWdhdGVfMSh0aGlzLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgYXR0YWNoT3JpZ2luVG9QYXRjaGVkKHByb3RvW25hbWVdLCBkZWxlZ2F0ZSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGRlbGVnYXRlO1xuICAgIH1cbiAgICAvLyBUT0RPOiBASmlhTGlQYXNzaW9uLCBzdXBwb3J0IGNhbmNlbCB0YXNrIGxhdGVyIGlmIG5lY2Vzc2FyeVxuICAgIGZ1bmN0aW9uIHBhdGNoTWFjcm9UYXNrKG9iaiwgZnVuY05hbWUsIG1ldGFDcmVhdG9yKSB7XG4gICAgICAgIHZhciBzZXROYXRpdmUgPSBudWxsO1xuICAgICAgICBmdW5jdGlvbiBzY2hlZHVsZVRhc2sodGFzaykge1xuICAgICAgICAgICAgdmFyIGRhdGEgPSB0YXNrLmRhdGE7XG4gICAgICAgICAgICBkYXRhLmFyZ3NbZGF0YS5jYklkeF0gPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgdGFzay5pbnZva2UuYXBwbHkodGhpcywgYXJndW1lbnRzKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBzZXROYXRpdmUuYXBwbHkoZGF0YS50YXJnZXQsIGRhdGEuYXJncyk7XG4gICAgICAgICAgICByZXR1cm4gdGFzaztcbiAgICAgICAgfVxuICAgICAgICBzZXROYXRpdmUgPSBwYXRjaE1ldGhvZChvYmosIGZ1bmNOYW1lLCBmdW5jdGlvbiAoZGVsZWdhdGUpIHsgcmV0dXJuIGZ1bmN0aW9uIChzZWxmLCBhcmdzKSB7XG4gICAgICAgICAgICB2YXIgbWV0YSA9IG1ldGFDcmVhdG9yKHNlbGYsIGFyZ3MpO1xuICAgICAgICAgICAgaWYgKG1ldGEuY2JJZHggPj0gMCAmJiB0eXBlb2YgYXJnc1ttZXRhLmNiSWR4XSA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgIHJldHVybiBzY2hlZHVsZU1hY3JvVGFza1dpdGhDdXJyZW50Wm9uZShtZXRhLm5hbWUsIGFyZ3NbbWV0YS5jYklkeF0sIG1ldGEsIHNjaGVkdWxlVGFzayk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyBjYXVzZSBhbiBlcnJvciBieSBjYWxsaW5nIGl0IGRpcmVjdGx5LlxuICAgICAgICAgICAgICAgIHJldHVybiBkZWxlZ2F0ZS5hcHBseShzZWxmLCBhcmdzKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTsgfSk7XG4gICAgfVxuICAgIGZ1bmN0aW9uIGF0dGFjaE9yaWdpblRvUGF0Y2hlZChwYXRjaGVkLCBvcmlnaW5hbCkge1xuICAgICAgICBwYXRjaGVkW3pvbmVTeW1ib2wkMSgnT3JpZ2luYWxEZWxlZ2F0ZScpXSA9IG9yaWdpbmFsO1xuICAgIH1cbiAgICB2YXIgaXNEZXRlY3RlZElFT3JFZGdlID0gZmFsc2U7XG4gICAgdmFyIGllT3JFZGdlID0gZmFsc2U7XG4gICAgZnVuY3Rpb24gaXNJRSgpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgIHZhciB1YSA9IGludGVybmFsV2luZG93Lm5hdmlnYXRvci51c2VyQWdlbnQ7XG4gICAgICAgICAgICBpZiAodWEuaW5kZXhPZignTVNJRSAnKSAhPT0gLTEgfHwgdWEuaW5kZXhPZignVHJpZGVudC8nKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGZ1bmN0aW9uIGlzSUVPckVkZ2UoKSB7XG4gICAgICAgIGlmIChpc0RldGVjdGVkSUVPckVkZ2UpIHtcbiAgICAgICAgICAgIHJldHVybiBpZU9yRWRnZTtcbiAgICAgICAgfVxuICAgICAgICBpc0RldGVjdGVkSUVPckVkZ2UgPSB0cnVlO1xuICAgICAgICB0cnkge1xuICAgICAgICAgICAgdmFyIHVhID0gaW50ZXJuYWxXaW5kb3cubmF2aWdhdG9yLnVzZXJBZ2VudDtcbiAgICAgICAgICAgIGlmICh1YS5pbmRleE9mKCdNU0lFICcpICE9PSAtMSB8fCB1YS5pbmRleE9mKCdUcmlkZW50LycpICE9PSAtMSB8fCB1YS5pbmRleE9mKCdFZGdlLycpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgIGllT3JFZGdlID0gdHJ1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gaWVPckVkZ2U7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdab25lQXdhcmVQcm9taXNlJywgZnVuY3Rpb24gKGdsb2JhbCwgWm9uZSwgYXBpKSB7XG4gICAgICAgIHZhciBPYmplY3RHZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3IgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yO1xuICAgICAgICB2YXIgT2JqZWN0RGVmaW5lUHJvcGVydHkgPSBPYmplY3QuZGVmaW5lUHJvcGVydHk7XG4gICAgICAgIGZ1bmN0aW9uIHJlYWRhYmxlT2JqZWN0VG9TdHJpbmcob2JqKSB7XG4gICAgICAgICAgICBpZiAob2JqICYmIG9iai50b1N0cmluZyA9PT0gT2JqZWN0LnByb3RvdHlwZS50b1N0cmluZykge1xuICAgICAgICAgICAgICAgIHZhciBjbGFzc05hbWUgPSBvYmouY29uc3RydWN0b3IgJiYgb2JqLmNvbnN0cnVjdG9yLm5hbWU7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChjbGFzc05hbWUgPyBjbGFzc05hbWUgOiAnJykgKyAnOiAnICsgSlNPTi5zdHJpbmdpZnkob2JqKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBvYmogPyBvYmoudG9TdHJpbmcoKSA6IE9iamVjdC5wcm90b3R5cGUudG9TdHJpbmcuY2FsbChvYmopO1xuICAgICAgICB9XG4gICAgICAgIHZhciBfX3N5bWJvbF9fID0gYXBpLnN5bWJvbDtcbiAgICAgICAgdmFyIF91bmNhdWdodFByb21pc2VFcnJvcnMgPSBbXTtcbiAgICAgICAgdmFyIGlzRGlzYWJsZVdyYXBwaW5nVW5jYXVnaHRQcm9taXNlUmVqZWN0aW9uID0gZ2xvYmFsW19fc3ltYm9sX18oJ0RJU0FCTEVfV1JBUFBJTkdfVU5DQVVHSFRfUFJPTUlTRV9SRUpFQ1RJT04nKV0gPT09IHRydWU7XG4gICAgICAgIHZhciBzeW1ib2xQcm9taXNlID0gX19zeW1ib2xfXygnUHJvbWlzZScpO1xuICAgICAgICB2YXIgc3ltYm9sVGhlbiA9IF9fc3ltYm9sX18oJ3RoZW4nKTtcbiAgICAgICAgdmFyIGNyZWF0aW9uVHJhY2UgPSAnX19jcmVhdGlvblRyYWNlX18nO1xuICAgICAgICBhcGkub25VbmhhbmRsZWRFcnJvciA9IGZ1bmN0aW9uIChlKSB7XG4gICAgICAgICAgICBpZiAoYXBpLnNob3dVbmNhdWdodEVycm9yKCkpIHtcbiAgICAgICAgICAgICAgICB2YXIgcmVqZWN0aW9uID0gZSAmJiBlLnJlamVjdGlvbjtcbiAgICAgICAgICAgICAgICBpZiAocmVqZWN0aW9uKSB7XG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ1VuaGFuZGxlZCBQcm9taXNlIHJlamVjdGlvbjonLCByZWplY3Rpb24gaW5zdGFuY2VvZiBFcnJvciA/IHJlamVjdGlvbi5tZXNzYWdlIDogcmVqZWN0aW9uLCAnOyBab25lOicsIGUuem9uZS5uYW1lLCAnOyBUYXNrOicsIGUudGFzayAmJiBlLnRhc2suc291cmNlLCAnOyBWYWx1ZTonLCByZWplY3Rpb24sIHJlamVjdGlvbiBpbnN0YW5jZW9mIEVycm9yID8gcmVqZWN0aW9uLnN0YWNrIDogdW5kZWZpbmVkKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoZSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9O1xuICAgICAgICBhcGkubWljcm90YXNrRHJhaW5Eb25lID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgdmFyIF9sb29wXzIgPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgdmFyIHVuY2F1Z2h0UHJvbWlzZUVycm9yID0gX3VuY2F1Z2h0UHJvbWlzZUVycm9ycy5zaGlmdCgpO1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIHVuY2F1Z2h0UHJvbWlzZUVycm9yLnpvbmUucnVuR3VhcmRlZChmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodW5jYXVnaHRQcm9taXNlRXJyb3IudGhyb3dPcmlnaW5hbCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IHVuY2F1Z2h0UHJvbWlzZUVycm9yLnJlamVjdGlvbjtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IHVuY2F1Z2h0UHJvbWlzZUVycm9yO1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgIGhhbmRsZVVuaGFuZGxlZFJlamVjdGlvbihlcnJvcik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHdoaWxlIChfdW5jYXVnaHRQcm9taXNlRXJyb3JzLmxlbmd0aCkge1xuICAgICAgICAgICAgICAgIF9sb29wXzIoKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfTtcbiAgICAgICAgdmFyIFVOSEFORExFRF9QUk9NSVNFX1JFSkVDVElPTl9IQU5ETEVSX1NZTUJPTCA9IF9fc3ltYm9sX18oJ3VuaGFuZGxlZFByb21pc2VSZWplY3Rpb25IYW5kbGVyJyk7XG4gICAgICAgIGZ1bmN0aW9uIGhhbmRsZVVuaGFuZGxlZFJlamVjdGlvbihlKSB7XG4gICAgICAgICAgICBhcGkub25VbmhhbmRsZWRFcnJvcihlKTtcbiAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgdmFyIGhhbmRsZXIgPSBab25lW1VOSEFORExFRF9QUk9NSVNFX1JFSkVDVElPTl9IQU5ETEVSX1NZTUJPTF07XG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZiBoYW5kbGVyID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgIGhhbmRsZXIuY2FsbCh0aGlzLCBlKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgZnVuY3Rpb24gaXNUaGVuYWJsZSh2YWx1ZSkge1xuICAgICAgICAgICAgcmV0dXJuIHZhbHVlICYmIHZhbHVlLnRoZW47XG4gICAgICAgIH1cbiAgICAgICAgZnVuY3Rpb24gZm9yd2FyZFJlc29sdXRpb24odmFsdWUpIHtcbiAgICAgICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgICAgfVxuICAgICAgICBmdW5jdGlvbiBmb3J3YXJkUmVqZWN0aW9uKHJlamVjdGlvbikge1xuICAgICAgICAgICAgcmV0dXJuIFpvbmVBd2FyZVByb21pc2UucmVqZWN0KHJlamVjdGlvbik7XG4gICAgICAgIH1cbiAgICAgICAgdmFyIHN5bWJvbFN0YXRlID0gX19zeW1ib2xfXygnc3RhdGUnKTtcbiAgICAgICAgdmFyIHN5bWJvbFZhbHVlID0gX19zeW1ib2xfXygndmFsdWUnKTtcbiAgICAgICAgdmFyIHN5bWJvbEZpbmFsbHkgPSBfX3N5bWJvbF9fKCdmaW5hbGx5Jyk7XG4gICAgICAgIHZhciBzeW1ib2xQYXJlbnRQcm9taXNlVmFsdWUgPSBfX3N5bWJvbF9fKCdwYXJlbnRQcm9taXNlVmFsdWUnKTtcbiAgICAgICAgdmFyIHN5bWJvbFBhcmVudFByb21pc2VTdGF0ZSA9IF9fc3ltYm9sX18oJ3BhcmVudFByb21pc2VTdGF0ZScpO1xuICAgICAgICB2YXIgc291cmNlID0gJ1Byb21pc2UudGhlbic7XG4gICAgICAgIHZhciBVTlJFU09MVkVEID0gbnVsbDtcbiAgICAgICAgdmFyIFJFU09MVkVEID0gdHJ1ZTtcbiAgICAgICAgdmFyIFJFSkVDVEVEID0gZmFsc2U7XG4gICAgICAgIHZhciBSRUpFQ1RFRF9OT19DQVRDSCA9IDA7XG4gICAgICAgIGZ1bmN0aW9uIG1ha2VSZXNvbHZlcihwcm9taXNlLCBzdGF0ZSkge1xuICAgICAgICAgICAgcmV0dXJuIGZ1bmN0aW9uICh2KSB7XG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZVByb21pc2UocHJvbWlzZSwgc3RhdGUsIHYpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICAgICAgICAgIHJlc29sdmVQcm9taXNlKHByb21pc2UsIGZhbHNlLCBlcnIpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAvLyBEbyBub3QgcmV0dXJuIHZhbHVlIG9yIHlvdSB3aWxsIGJyZWFrIHRoZSBQcm9taXNlIHNwZWMuXG4gICAgICAgICAgICB9O1xuICAgICAgICB9XG4gICAgICAgIHZhciBvbmNlID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgdmFyIHdhc0NhbGxlZCA9IGZhbHNlO1xuICAgICAgICAgICAgcmV0dXJuIGZ1bmN0aW9uIHdyYXBwZXIod3JhcHBlZEZ1bmN0aW9uKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHdhc0NhbGxlZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHdhc0NhbGxlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgIHdyYXBwZWRGdW5jdGlvbi5hcHBseShudWxsLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICB9O1xuICAgICAgICB9O1xuICAgICAgICB2YXIgVFlQRV9FUlJPUiA9ICdQcm9taXNlIHJlc29sdmVkIHdpdGggaXRzZWxmJztcbiAgICAgICAgdmFyIENVUlJFTlRfVEFTS19UUkFDRV9TWU1CT0wgPSBfX3N5bWJvbF9fKCdjdXJyZW50VGFza1RyYWNlJyk7XG4gICAgICAgIC8vIFByb21pc2UgUmVzb2x1dGlvblxuICAgICAgICBmdW5jdGlvbiByZXNvbHZlUHJvbWlzZShwcm9taXNlLCBzdGF0ZSwgdmFsdWUpIHtcbiAgICAgICAgICAgIHZhciBvbmNlV3JhcHBlciA9IG9uY2UoKTtcbiAgICAgICAgICAgIGlmIChwcm9taXNlID09PSB2YWx1ZSkge1xuICAgICAgICAgICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoVFlQRV9FUlJPUik7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAocHJvbWlzZVtzeW1ib2xTdGF0ZV0gPT09IFVOUkVTT0xWRUQpIHtcbiAgICAgICAgICAgICAgICAvLyBzaG91bGQgb25seSBnZXQgdmFsdWUudGhlbiBvbmNlIGJhc2VkIG9uIHByb21pc2Ugc3BlYy5cbiAgICAgICAgICAgICAgICB2YXIgdGhlbiA9IG51bGw7XG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGVvZiB2YWx1ZSA9PT0gJ29iamVjdCcgfHwgdHlwZW9mIHZhbHVlID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGVuID0gdmFsdWUgJiYgdmFsdWUudGhlbjtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICAgICAgICAgIG9uY2VXcmFwcGVyKGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlc29sdmVQcm9taXNlKHByb21pc2UsIGZhbHNlLCBlcnIpO1xuICAgICAgICAgICAgICAgICAgICB9KSgpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gcHJvbWlzZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgLy8gaWYgKHZhbHVlIGluc3RhbmNlb2YgWm9uZUF3YXJlUHJvbWlzZSkge1xuICAgICAgICAgICAgICAgIGlmIChzdGF0ZSAhPT0gUkVKRUNURUQgJiYgdmFsdWUgaW5zdGFuY2VvZiBab25lQXdhcmVQcm9taXNlICYmXG4gICAgICAgICAgICAgICAgICAgIHZhbHVlLmhhc093blByb3BlcnR5KHN5bWJvbFN0YXRlKSAmJiB2YWx1ZS5oYXNPd25Qcm9wZXJ0eShzeW1ib2xWYWx1ZSkgJiZcbiAgICAgICAgICAgICAgICAgICAgdmFsdWVbc3ltYm9sU3RhdGVdICE9PSBVTlJFU09MVkVEKSB7XG4gICAgICAgICAgICAgICAgICAgIGNsZWFyUmVqZWN0ZWROb0NhdGNoKHZhbHVlKTtcbiAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZVByb21pc2UocHJvbWlzZSwgdmFsdWVbc3ltYm9sU3RhdGVdLCB2YWx1ZVtzeW1ib2xWYWx1ZV0pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBlbHNlIGlmIChzdGF0ZSAhPT0gUkVKRUNURUQgJiYgdHlwZW9mIHRoZW4gPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoZW4uY2FsbCh2YWx1ZSwgb25jZVdyYXBwZXIobWFrZVJlc29sdmVyKHByb21pc2UsIHN0YXRlKSksIG9uY2VXcmFwcGVyKG1ha2VSZXNvbHZlcihwcm9taXNlLCBmYWxzZSkpKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBvbmNlV3JhcHBlcihmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZVByb21pc2UocHJvbWlzZSwgZmFsc2UsIGVycik7XG4gICAgICAgICAgICAgICAgICAgICAgICB9KSgpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBwcm9taXNlW3N5bWJvbFN0YXRlXSA9IHN0YXRlO1xuICAgICAgICAgICAgICAgICAgICB2YXIgcXVldWUgPSBwcm9taXNlW3N5bWJvbFZhbHVlXTtcbiAgICAgICAgICAgICAgICAgICAgcHJvbWlzZVtzeW1ib2xWYWx1ZV0gPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb21pc2Vbc3ltYm9sRmluYWxseV0gPT09IHN5bWJvbEZpbmFsbHkpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHRoZSBwcm9taXNlIGlzIGdlbmVyYXRlZCBieSBQcm9taXNlLnByb3RvdHlwZS5maW5hbGx5XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoc3RhdGUgPT09IFJFU09MVkVEKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gdGhlIHN0YXRlIGlzIHJlc29sdmVkLCBzaG91bGQgaWdub3JlIHRoZSB2YWx1ZVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGFuZCB1c2UgcGFyZW50IHByb21pc2UgdmFsdWVcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9taXNlW3N5bWJvbFN0YXRlXSA9IHByb21pc2Vbc3ltYm9sUGFyZW50UHJvbWlzZVN0YXRlXTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9taXNlW3N5bWJvbFZhbHVlXSA9IHByb21pc2Vbc3ltYm9sUGFyZW50UHJvbWlzZVZhbHVlXTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAvLyByZWNvcmQgdGFzayBpbmZvcm1hdGlvbiBpbiB2YWx1ZSB3aGVuIGVycm9yIG9jY3Vycywgc28gd2UgY2FuXG4gICAgICAgICAgICAgICAgICAgIC8vIGRvIHNvbWUgYWRkaXRpb25hbCB3b3JrIHN1Y2ggYXMgcmVuZGVyIGxvbmdTdGFja1RyYWNlXG4gICAgICAgICAgICAgICAgICAgIGlmIChzdGF0ZSA9PT0gUkVKRUNURUQgJiYgdmFsdWUgaW5zdGFuY2VvZiBFcnJvcikge1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gY2hlY2sgaWYgbG9uZ1N0YWNrVHJhY2Vab25lIGlzIGhlcmVcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0cmFjZSA9IFpvbmUuY3VycmVudFRhc2sgJiYgWm9uZS5jdXJyZW50VGFzay5kYXRhICYmXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgWm9uZS5jdXJyZW50VGFzay5kYXRhW2NyZWF0aW9uVHJhY2VdO1xuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHRyYWNlKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gb25seSBrZWVwIHRoZSBsb25nIHN0YWNrIHRyYWNlIGludG8gZXJyb3Igd2hlbiBpbiBsb25nU3RhY2tUcmFjZVpvbmVcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBPYmplY3REZWZpbmVQcm9wZXJ0eSh2YWx1ZSwgQ1VSUkVOVF9UQVNLX1RSQUNFX1NZTUJPTCwgeyBjb25maWd1cmFibGU6IHRydWUsIGVudW1lcmFibGU6IGZhbHNlLCB3cml0YWJsZTogdHJ1ZSwgdmFsdWU6IHRyYWNlIH0pO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcXVldWUubGVuZ3RoOykge1xuICAgICAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVSZXNvbHZlT3JSZWplY3QocHJvbWlzZSwgcXVldWVbaSsrXSwgcXVldWVbaSsrXSwgcXVldWVbaSsrXSwgcXVldWVbaSsrXSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgaWYgKHF1ZXVlLmxlbmd0aCA9PSAwICYmIHN0YXRlID09IFJFSkVDVEVEKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBwcm9taXNlW3N5bWJvbFN0YXRlXSA9IFJFSkVDVEVEX05PX0NBVENIO1xuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHVuY2F1Z2h0UHJvbWlzZUVycm9yID0gdmFsdWU7XG4gICAgICAgICAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIEhlcmUgd2UgdGhyb3dzIGEgbmV3IEVycm9yIHRvIHByaW50IG1vcmUgcmVhZGFibGUgZXJyb3IgbG9nXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gYW5kIGlmIHRoZSB2YWx1ZSBpcyBub3QgYW4gZXJyb3IsIHpvbmUuanMgYnVpbGRzIGFuIGBFcnJvcmBcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBPYmplY3QgaGVyZSB0byBhdHRhY2ggdGhlIHN0YWNrIGluZm9ybWF0aW9uLlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignVW5jYXVnaHQgKGluIHByb21pc2UpOiAnICsgcmVhZGFibGVPYmplY3RUb1N0cmluZyh2YWx1ZSkgK1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAodmFsdWUgJiYgdmFsdWUuc3RhY2sgPyAnXFxuJyArIHZhbHVlLnN0YWNrIDogJycpKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGNhdGNoIChlcnIpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB1bmNhdWdodFByb21pc2VFcnJvciA9IGVycjtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChpc0Rpc2FibGVXcmFwcGluZ1VuY2F1Z2h0UHJvbWlzZVJlamVjdGlvbikge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIElmIGRpc2FibGUgd3JhcHBpbmcgdW5jYXVnaHQgcHJvbWlzZSByZWplY3RcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyB1c2UgdGhlIHZhbHVlIGluc3RlYWQgb2Ygd3JhcHBpbmcgaXQuXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdW5jYXVnaHRQcm9taXNlRXJyb3IudGhyb3dPcmlnaW5hbCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB1bmNhdWdodFByb21pc2VFcnJvci5yZWplY3Rpb24gPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHVuY2F1Z2h0UHJvbWlzZUVycm9yLnByb21pc2UgPSBwcm9taXNlO1xuICAgICAgICAgICAgICAgICAgICAgICAgdW5jYXVnaHRQcm9taXNlRXJyb3Iuem9uZSA9IFpvbmUuY3VycmVudDtcbiAgICAgICAgICAgICAgICAgICAgICAgIHVuY2F1Z2h0UHJvbWlzZUVycm9yLnRhc2sgPSBab25lLmN1cnJlbnRUYXNrO1xuICAgICAgICAgICAgICAgICAgICAgICAgX3VuY2F1Z2h0UHJvbWlzZUVycm9ycy5wdXNoKHVuY2F1Z2h0UHJvbWlzZUVycm9yKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGFwaS5zY2hlZHVsZU1pY3JvVGFzaygpOyAvLyB0byBtYWtlIHN1cmUgdGhhdCBpdCBpcyBydW5uaW5nXG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICAvLyBSZXNvbHZpbmcgYW4gYWxyZWFkeSByZXNvbHZlZCBwcm9taXNlIGlzIGEgbm9vcC5cbiAgICAgICAgICAgIHJldHVybiBwcm9taXNlO1xuICAgICAgICB9XG4gICAgICAgIHZhciBSRUpFQ1RJT05fSEFORExFRF9IQU5ETEVSID0gX19zeW1ib2xfXygncmVqZWN0aW9uSGFuZGxlZEhhbmRsZXInKTtcbiAgICAgICAgZnVuY3Rpb24gY2xlYXJSZWplY3RlZE5vQ2F0Y2gocHJvbWlzZSkge1xuICAgICAgICAgICAgaWYgKHByb21pc2Vbc3ltYm9sU3RhdGVdID09PSBSRUpFQ1RFRF9OT19DQVRDSCkge1xuICAgICAgICAgICAgICAgIC8vIGlmIHRoZSBwcm9taXNlIGlzIHJlamVjdGVkIG5vIGNhdGNoIHN0YXR1c1xuICAgICAgICAgICAgICAgIC8vIGFuZCBxdWV1ZS5sZW5ndGggPiAwLCBtZWFucyB0aGVyZSBpcyBhIGVycm9yIGhhbmRsZXJcbiAgICAgICAgICAgICAgICAvLyBoZXJlIHRvIGhhbmRsZSB0aGUgcmVqZWN0ZWQgcHJvbWlzZSwgd2Ugc2hvdWxkIHRyaWdnZXJcbiAgICAgICAgICAgICAgICAvLyB3aW5kb3dzLnJlamVjdGlvbmhhbmRsZWQgZXZlbnRIYW5kbGVyIG9yIG5vZGVqcyByZWplY3Rpb25IYW5kbGVkXG4gICAgICAgICAgICAgICAgLy8gZXZlbnRIYW5kbGVyXG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIGhhbmRsZXIgPSBab25lW1JFSkVDVElPTl9IQU5ETEVEX0hBTkRMRVJdO1xuICAgICAgICAgICAgICAgICAgICBpZiAoaGFuZGxlciAmJiB0eXBlb2YgaGFuZGxlciA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgICAgICAgICAgaGFuZGxlci5jYWxsKHRoaXMsIHsgcmVqZWN0aW9uOiBwcm9taXNlW3N5bWJvbFZhbHVlXSwgcHJvbWlzZTogcHJvbWlzZSB9KTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHByb21pc2Vbc3ltYm9sU3RhdGVdID0gUkVKRUNURUQ7XG4gICAgICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBfdW5jYXVnaHRQcm9taXNlRXJyb3JzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9taXNlID09PSBfdW5jYXVnaHRQcm9taXNlRXJyb3JzW2ldLnByb21pc2UpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIF91bmNhdWdodFByb21pc2VFcnJvcnMuc3BsaWNlKGksIDEpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGZ1bmN0aW9uIHNjaGVkdWxlUmVzb2x2ZU9yUmVqZWN0KHByb21pc2UsIHpvbmUsIGNoYWluUHJvbWlzZSwgb25GdWxmaWxsZWQsIG9uUmVqZWN0ZWQpIHtcbiAgICAgICAgICAgIGNsZWFyUmVqZWN0ZWROb0NhdGNoKHByb21pc2UpO1xuICAgICAgICAgICAgdmFyIHByb21pc2VTdGF0ZSA9IHByb21pc2Vbc3ltYm9sU3RhdGVdO1xuICAgICAgICAgICAgdmFyIGRlbGVnYXRlID0gcHJvbWlzZVN0YXRlID9cbiAgICAgICAgICAgICAgICAodHlwZW9mIG9uRnVsZmlsbGVkID09PSAnZnVuY3Rpb24nKSA/IG9uRnVsZmlsbGVkIDogZm9yd2FyZFJlc29sdXRpb24gOlxuICAgICAgICAgICAgICAgICh0eXBlb2Ygb25SZWplY3RlZCA9PT0gJ2Z1bmN0aW9uJykgPyBvblJlamVjdGVkIDpcbiAgICAgICAgICAgICAgICAgICAgZm9yd2FyZFJlamVjdGlvbjtcbiAgICAgICAgICAgIHpvbmUuc2NoZWR1bGVNaWNyb1Rhc2soc291cmNlLCBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHBhcmVudFByb21pc2VWYWx1ZSA9IHByb21pc2Vbc3ltYm9sVmFsdWVdO1xuICAgICAgICAgICAgICAgICAgICB2YXIgaXNGaW5hbGx5UHJvbWlzZSA9ICEhY2hhaW5Qcm9taXNlICYmIHN5bWJvbEZpbmFsbHkgPT09IGNoYWluUHJvbWlzZVtzeW1ib2xGaW5hbGx5XTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKGlzRmluYWxseVByb21pc2UpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGlmIHRoZSBwcm9taXNlIGlzIGdlbmVyYXRlZCBmcm9tIGZpbmFsbHkgY2FsbCwga2VlcCBwYXJlbnQgcHJvbWlzZSdzIHN0YXRlIGFuZCB2YWx1ZVxuICAgICAgICAgICAgICAgICAgICAgICAgY2hhaW5Qcm9taXNlW3N5bWJvbFBhcmVudFByb21pc2VWYWx1ZV0gPSBwYXJlbnRQcm9taXNlVmFsdWU7XG4gICAgICAgICAgICAgICAgICAgICAgICBjaGFpblByb21pc2Vbc3ltYm9sUGFyZW50UHJvbWlzZVN0YXRlXSA9IHByb21pc2VTdGF0ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAvLyBzaG91bGQgbm90IHBhc3MgdmFsdWUgdG8gZmluYWxseSBjYWxsYmFja1xuICAgICAgICAgICAgICAgICAgICB2YXIgdmFsdWUgPSB6b25lLnJ1bihkZWxlZ2F0ZSwgdW5kZWZpbmVkLCBpc0ZpbmFsbHlQcm9taXNlICYmIGRlbGVnYXRlICE9PSBmb3J3YXJkUmVqZWN0aW9uICYmIGRlbGVnYXRlICE9PSBmb3J3YXJkUmVzb2x1dGlvbiA/XG4gICAgICAgICAgICAgICAgICAgICAgICBbXSA6XG4gICAgICAgICAgICAgICAgICAgICAgICBbcGFyZW50UHJvbWlzZVZhbHVlXSk7XG4gICAgICAgICAgICAgICAgICAgIHJlc29sdmVQcm9taXNlKGNoYWluUHJvbWlzZSwgdHJ1ZSwgdmFsdWUpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gaWYgZXJyb3Igb2NjdXJzLCBzaG91bGQgYWx3YXlzIHJldHVybiB0aGlzIGVycm9yXG4gICAgICAgICAgICAgICAgICAgIHJlc29sdmVQcm9taXNlKGNoYWluUHJvbWlzZSwgZmFsc2UsIGVycm9yKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LCBjaGFpblByb21pc2UpO1xuICAgICAgICB9XG4gICAgICAgIHZhciBaT05FX0FXQVJFX1BST01JU0VfVE9fU1RSSU5HID0gJ2Z1bmN0aW9uIFpvbmVBd2FyZVByb21pc2UoKSB7IFtuYXRpdmUgY29kZV0gfSc7XG4gICAgICAgIHZhciBub29wID0gZnVuY3Rpb24gKCkgeyB9O1xuICAgICAgICB2YXIgQWdncmVnYXRlRXJyb3IgPSBnbG9iYWwuQWdncmVnYXRlRXJyb3I7XG4gICAgICAgIHZhciBab25lQXdhcmVQcm9taXNlID0gLyoqIEBjbGFzcyAqLyAoZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgZnVuY3Rpb24gWm9uZUF3YXJlUHJvbWlzZShleGVjdXRvcikge1xuICAgICAgICAgICAgICAgIHZhciBwcm9taXNlID0gdGhpcztcbiAgICAgICAgICAgICAgICBpZiAoIShwcm9taXNlIGluc3RhbmNlb2YgWm9uZUF3YXJlUHJvbWlzZSkpIHtcbiAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdNdXN0IGJlIGFuIGluc3RhbmNlb2YgUHJvbWlzZS4nKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcHJvbWlzZVtzeW1ib2xTdGF0ZV0gPSBVTlJFU09MVkVEO1xuICAgICAgICAgICAgICAgIHByb21pc2Vbc3ltYm9sVmFsdWVdID0gW107IC8vIHF1ZXVlO1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciBvbmNlV3JhcHBlciA9IG9uY2UoKTtcbiAgICAgICAgICAgICAgICAgICAgZXhlY3V0b3IgJiZcbiAgICAgICAgICAgICAgICAgICAgICAgIGV4ZWN1dG9yKG9uY2VXcmFwcGVyKG1ha2VSZXNvbHZlcihwcm9taXNlLCBSRVNPTFZFRCkpLCBvbmNlV3JhcHBlcihtYWtlUmVzb2x2ZXIocHJvbWlzZSwgUkVKRUNURUQpKSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGNhdGNoIChlcnJvcikge1xuICAgICAgICAgICAgICAgICAgICByZXNvbHZlUHJvbWlzZShwcm9taXNlLCBmYWxzZSwgZXJyb3IpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UudG9TdHJpbmcgPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIFpPTkVfQVdBUkVfUFJPTUlTRV9UT19TVFJJTkc7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgWm9uZUF3YXJlUHJvbWlzZS5yZXNvbHZlID0gZnVuY3Rpb24gKHZhbHVlKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHJlc29sdmVQcm9taXNlKG5ldyB0aGlzKG51bGwpLCBSRVNPTFZFRCwgdmFsdWUpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UucmVqZWN0ID0gZnVuY3Rpb24gKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHJlc29sdmVQcm9taXNlKG5ldyB0aGlzKG51bGwpLCBSRUpFQ1RFRCwgZXJyb3IpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UuYW55ID0gZnVuY3Rpb24gKHZhbHVlcykge1xuICAgICAgICAgICAgICAgIGlmICghdmFsdWVzIHx8IHR5cGVvZiB2YWx1ZXNbU3ltYm9sLml0ZXJhdG9yXSAhPT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gUHJvbWlzZS5yZWplY3QobmV3IEFnZ3JlZ2F0ZUVycm9yKFtdLCAnQWxsIHByb21pc2VzIHdlcmUgcmVqZWN0ZWQnKSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHZhciBwcm9taXNlcyA9IFtdO1xuICAgICAgICAgICAgICAgIHZhciBjb3VudCA9IDA7XG4gICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgX2kgPSAwLCB2YWx1ZXNfMSA9IHZhbHVlczsgX2kgPCB2YWx1ZXNfMS5sZW5ndGg7IF9pKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciB2ID0gdmFsdWVzXzFbX2ldO1xuICAgICAgICAgICAgICAgICAgICAgICAgY291bnQrKztcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb21pc2VzLnB1c2goWm9uZUF3YXJlUHJvbWlzZS5yZXNvbHZlKHYpKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBQcm9taXNlLnJlamVjdChuZXcgQWdncmVnYXRlRXJyb3IoW10sICdBbGwgcHJvbWlzZXMgd2VyZSByZWplY3RlZCcpKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKGNvdW50ID09PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBQcm9taXNlLnJlamVjdChuZXcgQWdncmVnYXRlRXJyb3IoW10sICdBbGwgcHJvbWlzZXMgd2VyZSByZWplY3RlZCcpKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdmFyIGZpbmlzaGVkID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgdmFyIGVycm9ycyA9IFtdO1xuICAgICAgICAgICAgICAgIHJldHVybiBuZXcgWm9uZUF3YXJlUHJvbWlzZShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7XG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcHJvbWlzZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb21pc2VzW2ldLnRoZW4oZnVuY3Rpb24gKHYpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZmluaXNoZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmaW5pc2hlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZSh2KTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH0sIGZ1bmN0aW9uIChlcnIpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlcnJvcnMucHVzaChlcnIpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50LS07XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGNvdW50ID09PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZpbmlzaGVkID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVqZWN0KG5ldyBBZ2dyZWdhdGVFcnJvcihlcnJvcnMsICdBbGwgcHJvbWlzZXMgd2VyZSByZWplY3RlZCcpKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIDtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UucmFjZSA9IGZ1bmN0aW9uICh2YWx1ZXMpIHtcbiAgICAgICAgICAgICAgICB2YXIgcmVzb2x2ZTtcbiAgICAgICAgICAgICAgICB2YXIgcmVqZWN0O1xuICAgICAgICAgICAgICAgIHZhciBwcm9taXNlID0gbmV3IHRoaXMoZnVuY3Rpb24gKHJlcywgcmVqKSB7XG4gICAgICAgICAgICAgICAgICAgIHJlc29sdmUgPSByZXM7XG4gICAgICAgICAgICAgICAgICAgIHJlamVjdCA9IHJlajtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICBmdW5jdGlvbiBvblJlc29sdmUodmFsdWUpIHtcbiAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZSh2YWx1ZSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGZ1bmN0aW9uIG9uUmVqZWN0KGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgIHJlamVjdChlcnJvcik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGZvciAodmFyIF9pID0gMCwgdmFsdWVzXzIgPSB2YWx1ZXM7IF9pIDwgdmFsdWVzXzIubGVuZ3RoOyBfaSsrKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciB2YWx1ZSA9IHZhbHVlc18yW19pXTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKCFpc1RoZW5hYmxlKHZhbHVlKSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgdmFsdWUgPSB0aGlzLnJlc29sdmUodmFsdWUpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHZhbHVlLnRoZW4ob25SZXNvbHZlLCBvblJlamVjdCk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBwcm9taXNlO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UuYWxsID0gZnVuY3Rpb24gKHZhbHVlcykge1xuICAgICAgICAgICAgICAgIHJldHVybiBab25lQXdhcmVQcm9taXNlLmFsbFdpdGhDYWxsYmFjayh2YWx1ZXMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UuYWxsU2V0dGxlZCA9IGZ1bmN0aW9uICh2YWx1ZXMpIHtcbiAgICAgICAgICAgICAgICB2YXIgUCA9IHRoaXMgJiYgdGhpcy5wcm90b3R5cGUgaW5zdGFuY2VvZiBab25lQXdhcmVQcm9taXNlID8gdGhpcyA6IFpvbmVBd2FyZVByb21pc2U7XG4gICAgICAgICAgICAgICAgcmV0dXJuIFAuYWxsV2l0aENhbGxiYWNrKHZhbHVlcywge1xuICAgICAgICAgICAgICAgICAgICB0aGVuQ2FsbGJhY2s6IGZ1bmN0aW9uICh2YWx1ZSkgeyByZXR1cm4gKHsgc3RhdHVzOiAnZnVsZmlsbGVkJywgdmFsdWU6IHZhbHVlIH0pOyB9LFxuICAgICAgICAgICAgICAgICAgICBlcnJvckNhbGxiYWNrOiBmdW5jdGlvbiAoZXJyKSB7IHJldHVybiAoeyBzdGF0dXM6ICdyZWplY3RlZCcsIHJlYXNvbjogZXJyIH0pOyB9XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgWm9uZUF3YXJlUHJvbWlzZS5hbGxXaXRoQ2FsbGJhY2sgPSBmdW5jdGlvbiAodmFsdWVzLCBjYWxsYmFjaykge1xuICAgICAgICAgICAgICAgIHZhciByZXNvbHZlO1xuICAgICAgICAgICAgICAgIHZhciByZWplY3Q7XG4gICAgICAgICAgICAgICAgdmFyIHByb21pc2UgPSBuZXcgdGhpcyhmdW5jdGlvbiAocmVzLCByZWopIHtcbiAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZSA9IHJlcztcbiAgICAgICAgICAgICAgICAgICAgcmVqZWN0ID0gcmVqO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIC8vIFN0YXJ0IGF0IDIgdG8gcHJldmVudCBwcmVtYXR1cmVseSByZXNvbHZpbmcgaWYgLnRoZW4gaXMgY2FsbGVkIGltbWVkaWF0ZWx5LlxuICAgICAgICAgICAgICAgIHZhciB1bnJlc29sdmVkQ291bnQgPSAyO1xuICAgICAgICAgICAgICAgIHZhciB2YWx1ZUluZGV4ID0gMDtcbiAgICAgICAgICAgICAgICB2YXIgcmVzb2x2ZWRWYWx1ZXMgPSBbXTtcbiAgICAgICAgICAgICAgICB2YXIgX2xvb3BfMyA9IGZ1bmN0aW9uICh2YWx1ZSkge1xuICAgICAgICAgICAgICAgICAgICBpZiAoIWlzVGhlbmFibGUodmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9IHRoaXNfMS5yZXNvbHZlKHZhbHVlKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YXIgY3VyVmFsdWVJbmRleCA9IHZhbHVlSW5kZXg7XG4gICAgICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZS50aGVuKGZ1bmN0aW9uICh2YWx1ZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc29sdmVkVmFsdWVzW2N1clZhbHVlSW5kZXhdID0gY2FsbGJhY2sgPyBjYWxsYmFjay50aGVuQ2FsbGJhY2sodmFsdWUpIDogdmFsdWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdW5yZXNvbHZlZENvdW50LS07XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHVucmVzb2x2ZWRDb3VudCA9PT0gMCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNvbHZlKHJlc29sdmVkVmFsdWVzKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9LCBmdW5jdGlvbiAoZXJyKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCFjYWxsYmFjaykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZWplY3QoZXJyKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc29sdmVkVmFsdWVzW2N1clZhbHVlSW5kZXhdID0gY2FsbGJhY2suZXJyb3JDYWxsYmFjayhlcnIpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB1bnJlc29sdmVkQ291bnQtLTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHVucmVzb2x2ZWRDb3VudCA9PT0gMCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzb2x2ZShyZXNvbHZlZFZhbHVlcyk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBjYXRjaCAodGhlbkVycikge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmVqZWN0KHRoZW5FcnIpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHVucmVzb2x2ZWRDb3VudCsrO1xuICAgICAgICAgICAgICAgICAgICB2YWx1ZUluZGV4Kys7XG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICB2YXIgdGhpc18xID0gdGhpcztcbiAgICAgICAgICAgICAgICBmb3IgKHZhciBfaSA9IDAsIHZhbHVlc18zID0gdmFsdWVzOyBfaSA8IHZhbHVlc18zLmxlbmd0aDsgX2krKykge1xuICAgICAgICAgICAgICAgICAgICB2YXIgdmFsdWUgPSB2YWx1ZXNfM1tfaV07XG4gICAgICAgICAgICAgICAgICAgIF9sb29wXzModmFsdWUpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAvLyBNYWtlIHRoZSB1bnJlc29sdmVkQ291bnQgemVyby1iYXNlZCBhZ2Fpbi5cbiAgICAgICAgICAgICAgICB1bnJlc29sdmVkQ291bnQgLT0gMjtcbiAgICAgICAgICAgICAgICBpZiAodW5yZXNvbHZlZENvdW50ID09PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgIHJlc29sdmUocmVzb2x2ZWRWYWx1ZXMpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gcHJvbWlzZTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkoWm9uZUF3YXJlUHJvbWlzZS5wcm90b3R5cGUsIFN5bWJvbC50b1N0cmluZ1RhZywge1xuICAgICAgICAgICAgICAgIGdldDogZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gJ1Byb21pc2UnO1xuICAgICAgICAgICAgICAgIH0sXG4gICAgICAgICAgICAgICAgZW51bWVyYWJsZTogZmFsc2UsXG4gICAgICAgICAgICAgICAgY29uZmlndXJhYmxlOiB0cnVlXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShab25lQXdhcmVQcm9taXNlLnByb3RvdHlwZSwgU3ltYm9sLnNwZWNpZXMsIHtcbiAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFpvbmVBd2FyZVByb21pc2U7XG4gICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICBlbnVtZXJhYmxlOiBmYWxzZSxcbiAgICAgICAgICAgICAgICBjb25maWd1cmFibGU6IHRydWVcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgWm9uZUF3YXJlUHJvbWlzZS5wcm90b3R5cGUudGhlbiA9IGZ1bmN0aW9uIChvbkZ1bGZpbGxlZCwgb25SZWplY3RlZCkge1xuICAgICAgICAgICAgICAgIHZhciBfYTtcbiAgICAgICAgICAgICAgICAvLyBXZSBtdXN0IHJlYWQgYFN5bWJvbC5zcGVjaWVzYCBzYWZlbHkgYmVjYXVzZSBgdGhpc2AgbWF5IGJlIGFueXRoaW5nLiBGb3IgaW5zdGFuY2UsIGB0aGlzYFxuICAgICAgICAgICAgICAgIC8vIG1heSBiZSBhbiBvYmplY3Qgd2l0aG91dCBhIHByb3RvdHlwZSAoY3JlYXRlZCB0aHJvdWdoIGBPYmplY3QuY3JlYXRlKG51bGwpYCk7IHRodXNcbiAgICAgICAgICAgICAgICAvLyBgdGhpcy5jb25zdHJ1Y3RvcmAgd2lsbCBiZSB1bmRlZmluZWQuIE9uZSBvZiB0aGUgdXNlIGNhc2VzIGlzIFN5c3RlbUpTIGNyZWF0aW5nXG4gICAgICAgICAgICAgICAgLy8gcHJvdG90eXBlLWxlc3Mgb2JqZWN0cyAobW9kdWxlcykgdmlhIGBPYmplY3QuY3JlYXRlKG51bGwpYC4gVGhlIFN5c3RlbUpTIGNyZWF0ZXMgYW4gZW1wdHlcbiAgICAgICAgICAgICAgICAvLyBvYmplY3QgYW5kIGNvcGllcyBwcm9taXNlIHByb3BlcnRpZXMgaW50byB0aGF0IG9iamVjdCAod2l0aGluIHRoZSBgZ2V0T3JDcmVhdGVMb2FkYFxuICAgICAgICAgICAgICAgIC8vIGZ1bmN0aW9uKS4gVGhlIHpvbmUuanMgdGhlbiBjaGVja3MgaWYgdGhlIHJlc29sdmVkIHZhbHVlIGhhcyB0aGUgYHRoZW5gIG1ldGhvZCBhbmQgaW52b2tlc1xuICAgICAgICAgICAgICAgIC8vIGl0IHdpdGggdGhlIGB2YWx1ZWAgY29udGV4dC4gT3RoZXJ3aXNlLCB0aGlzIHdpbGwgdGhyb3cgYW4gZXJyb3I6IGBUeXBlRXJyb3I6IENhbm5vdCByZWFkXG4gICAgICAgICAgICAgICAgLy8gcHJvcGVydGllcyBvZiB1bmRlZmluZWQgKHJlYWRpbmcgJ1N5bWJvbChTeW1ib2wuc3BlY2llcyknKWAuXG4gICAgICAgICAgICAgICAgdmFyIEMgPSAoX2EgPSB0aGlzLmNvbnN0cnVjdG9yKSA9PT0gbnVsbCB8fCBfYSA9PT0gdm9pZCAwID8gdm9pZCAwIDogX2FbU3ltYm9sLnNwZWNpZXNdO1xuICAgICAgICAgICAgICAgIGlmICghQyB8fCB0eXBlb2YgQyAhPT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgICAgICAgICAgICBDID0gdGhpcy5jb25zdHJ1Y3RvciB8fCBab25lQXdhcmVQcm9taXNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB2YXIgY2hhaW5Qcm9taXNlID0gbmV3IEMobm9vcCk7XG4gICAgICAgICAgICAgICAgdmFyIHpvbmUgPSBab25lLmN1cnJlbnQ7XG4gICAgICAgICAgICAgICAgaWYgKHRoaXNbc3ltYm9sU3RhdGVdID09IFVOUkVTT0xWRUQpIHtcbiAgICAgICAgICAgICAgICAgICAgdGhpc1tzeW1ib2xWYWx1ZV0ucHVzaCh6b25lLCBjaGFpblByb21pc2UsIG9uRnVsZmlsbGVkLCBvblJlamVjdGVkKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHNjaGVkdWxlUmVzb2x2ZU9yUmVqZWN0KHRoaXMsIHpvbmUsIGNoYWluUHJvbWlzZSwgb25GdWxmaWxsZWQsIG9uUmVqZWN0ZWQpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gY2hhaW5Qcm9taXNlO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UucHJvdG90eXBlLmNhdGNoID0gZnVuY3Rpb24gKG9uUmVqZWN0ZWQpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gdGhpcy50aGVuKG51bGwsIG9uUmVqZWN0ZWQpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIFpvbmVBd2FyZVByb21pc2UucHJvdG90eXBlLmZpbmFsbHkgPSBmdW5jdGlvbiAob25GaW5hbGx5KSB7XG4gICAgICAgICAgICAgICAgdmFyIF9hO1xuICAgICAgICAgICAgICAgIC8vIFNlZSBjb21tZW50IG9uIHRoZSBjYWxsIHRvIGB0aGVuYCBhYm91dCB3aHkgdGhlZSBgU3ltYm9sLnNwZWNpZXNgIGlzIHNhZmVseSBhY2Nlc3NlZC5cbiAgICAgICAgICAgICAgICB2YXIgQyA9IChfYSA9IHRoaXMuY29uc3RydWN0b3IpID09PSBudWxsIHx8IF9hID09PSB2b2lkIDAgPyB2b2lkIDAgOiBfYVtTeW1ib2wuc3BlY2llc107XG4gICAgICAgICAgICAgICAgaWYgKCFDIHx8IHR5cGVvZiBDICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgIEMgPSBab25lQXdhcmVQcm9taXNlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB2YXIgY2hhaW5Qcm9taXNlID0gbmV3IEMobm9vcCk7XG4gICAgICAgICAgICAgICAgY2hhaW5Qcm9taXNlW3N5bWJvbEZpbmFsbHldID0gc3ltYm9sRmluYWxseTtcbiAgICAgICAgICAgICAgICB2YXIgem9uZSA9IFpvbmUuY3VycmVudDtcbiAgICAgICAgICAgICAgICBpZiAodGhpc1tzeW1ib2xTdGF0ZV0gPT0gVU5SRVNPTFZFRCkge1xuICAgICAgICAgICAgICAgICAgICB0aGlzW3N5bWJvbFZhbHVlXS5wdXNoKHpvbmUsIGNoYWluUHJvbWlzZSwgb25GaW5hbGx5LCBvbkZpbmFsbHkpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgc2NoZWR1bGVSZXNvbHZlT3JSZWplY3QodGhpcywgem9uZSwgY2hhaW5Qcm9taXNlLCBvbkZpbmFsbHksIG9uRmluYWxseSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBjaGFpblByb21pc2U7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgcmV0dXJuIFpvbmVBd2FyZVByb21pc2U7XG4gICAgICAgIH0oKSk7XG4gICAgICAgIC8vIFByb3RlY3QgYWdhaW5zdCBhZ2dyZXNzaXZlIG9wdGltaXplcnMgZHJvcHBpbmcgc2VlbWluZ2x5IHVudXNlZCBwcm9wZXJ0aWVzLlxuICAgICAgICAvLyBFLmcuIENsb3N1cmUgQ29tcGlsZXIgaW4gYWR2YW5jZWQgbW9kZS5cbiAgICAgICAgWm9uZUF3YXJlUHJvbWlzZVsncmVzb2x2ZSddID0gWm9uZUF3YXJlUHJvbWlzZS5yZXNvbHZlO1xuICAgICAgICBab25lQXdhcmVQcm9taXNlWydyZWplY3QnXSA9IFpvbmVBd2FyZVByb21pc2UucmVqZWN0O1xuICAgICAgICBab25lQXdhcmVQcm9taXNlWydyYWNlJ10gPSBab25lQXdhcmVQcm9taXNlLnJhY2U7XG4gICAgICAgIFpvbmVBd2FyZVByb21pc2VbJ2FsbCddID0gWm9uZUF3YXJlUHJvbWlzZS5hbGw7XG4gICAgICAgIHZhciBOYXRpdmVQcm9taXNlID0gZ2xvYmFsW3N5bWJvbFByb21pc2VdID0gZ2xvYmFsWydQcm9taXNlJ107XG4gICAgICAgIGdsb2JhbFsnUHJvbWlzZSddID0gWm9uZUF3YXJlUHJvbWlzZTtcbiAgICAgICAgdmFyIHN5bWJvbFRoZW5QYXRjaGVkID0gX19zeW1ib2xfXygndGhlblBhdGNoZWQnKTtcbiAgICAgICAgZnVuY3Rpb24gcGF0Y2hUaGVuKEN0b3IpIHtcbiAgICAgICAgICAgIHZhciBwcm90byA9IEN0b3IucHJvdG90eXBlO1xuICAgICAgICAgICAgdmFyIHByb3AgPSBPYmplY3RHZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3IocHJvdG8sICd0aGVuJyk7XG4gICAgICAgICAgICBpZiAocHJvcCAmJiAocHJvcC53cml0YWJsZSA9PT0gZmFsc2UgfHwgIXByb3AuY29uZmlndXJhYmxlKSkge1xuICAgICAgICAgICAgICAgIC8vIGNoZWNrIEN0b3IucHJvdG90eXBlLnRoZW4gcHJvcGVydHlEZXNjcmlwdG9yIGlzIHdyaXRhYmxlIG9yIG5vdFxuICAgICAgICAgICAgICAgIC8vIGluIG1ldGVvciBlbnYsIHdyaXRhYmxlIGlzIGZhbHNlLCB3ZSBzaG91bGQgaWdub3JlIHN1Y2ggY2FzZVxuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBvcmlnaW5hbFRoZW4gPSBwcm90by50aGVuO1xuICAgICAgICAgICAgLy8gS2VlcCBhIHJlZmVyZW5jZSB0byB0aGUgb3JpZ2luYWwgbWV0aG9kLlxuICAgICAgICAgICAgcHJvdG9bc3ltYm9sVGhlbl0gPSBvcmlnaW5hbFRoZW47XG4gICAgICAgICAgICBDdG9yLnByb3RvdHlwZS50aGVuID0gZnVuY3Rpb24gKG9uUmVzb2x2ZSwgb25SZWplY3QpIHtcbiAgICAgICAgICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xuICAgICAgICAgICAgICAgIHZhciB3cmFwcGVkID0gbmV3IFpvbmVBd2FyZVByb21pc2UoZnVuY3Rpb24gKHJlc29sdmUsIHJlamVjdCkge1xuICAgICAgICAgICAgICAgICAgICBvcmlnaW5hbFRoZW4uY2FsbChfdGhpcywgcmVzb2x2ZSwgcmVqZWN0KTtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICByZXR1cm4gd3JhcHBlZC50aGVuKG9uUmVzb2x2ZSwgb25SZWplY3QpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIEN0b3Jbc3ltYm9sVGhlblBhdGNoZWRdID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgICBhcGkucGF0Y2hUaGVuID0gcGF0Y2hUaGVuO1xuICAgICAgICBmdW5jdGlvbiB6b25laWZ5KGZuKSB7XG4gICAgICAgICAgICByZXR1cm4gZnVuY3Rpb24gKHNlbGYsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICB2YXIgcmVzdWx0UHJvbWlzZSA9IGZuLmFwcGx5KHNlbGYsIGFyZ3MpO1xuICAgICAgICAgICAgICAgIGlmIChyZXN1bHRQcm9taXNlIGluc3RhbmNlb2YgWm9uZUF3YXJlUHJvbWlzZSkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gcmVzdWx0UHJvbWlzZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdmFyIGN0b3IgPSByZXN1bHRQcm9taXNlLmNvbnN0cnVjdG9yO1xuICAgICAgICAgICAgICAgIGlmICghY3RvcltzeW1ib2xUaGVuUGF0Y2hlZF0pIHtcbiAgICAgICAgICAgICAgICAgICAgcGF0Y2hUaGVuKGN0b3IpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gcmVzdWx0UHJvbWlzZTtcbiAgICAgICAgICAgIH07XG4gICAgICAgIH1cbiAgICAgICAgaWYgKE5hdGl2ZVByb21pc2UpIHtcbiAgICAgICAgICAgIHBhdGNoVGhlbihOYXRpdmVQcm9taXNlKTtcbiAgICAgICAgICAgIHBhdGNoTWV0aG9kKGdsb2JhbCwgJ2ZldGNoJywgZnVuY3Rpb24gKGRlbGVnYXRlKSB7IHJldHVybiB6b25laWZ5KGRlbGVnYXRlKTsgfSk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gVGhpcyBpcyBub3QgcGFydCBvZiBwdWJsaWMgQVBJLCBidXQgaXQgaXMgdXNlZnVsIGZvciB0ZXN0cywgc28gd2UgZXhwb3NlIGl0LlxuICAgICAgICBQcm9taXNlW1pvbmUuX19zeW1ib2xfXygndW5jYXVnaHRQcm9taXNlRXJyb3JzJyldID0gX3VuY2F1Z2h0UHJvbWlzZUVycm9ycztcbiAgICAgICAgcmV0dXJuIFpvbmVBd2FyZVByb21pc2U7XG4gICAgfSk7XG4gICAgLyoqXG4gICAgICogQGxpY2Vuc2VcbiAgICAgKiBDb3B5cmlnaHQgR29vZ2xlIExMQyBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICAgICAqXG4gICAgICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAgICAgKiBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIGF0IGh0dHBzOi8vYW5ndWxhci5pby9saWNlbnNlXG4gICAgICovXG4gICAgLy8gb3ZlcnJpZGUgRnVuY3Rpb24ucHJvdG90eXBlLnRvU3RyaW5nIHRvIG1ha2Ugem9uZS5qcyBwYXRjaGVkIGZ1bmN0aW9uXG4gICAgLy8gbG9vayBsaWtlIG5hdGl2ZSBmdW5jdGlvblxuICAgIFpvbmUuX19sb2FkX3BhdGNoKCd0b1N0cmluZycsIGZ1bmN0aW9uIChnbG9iYWwpIHtcbiAgICAgICAgLy8gcGF0Y2ggRnVuYy5wcm90b3R5cGUudG9TdHJpbmcgdG8gbGV0IHRoZW0gbG9vayBsaWtlIG5hdGl2ZVxuICAgICAgICB2YXIgb3JpZ2luYWxGdW5jdGlvblRvU3RyaW5nID0gRnVuY3Rpb24ucHJvdG90eXBlLnRvU3RyaW5nO1xuICAgICAgICB2YXIgT1JJR0lOQUxfREVMRUdBVEVfU1lNQk9MID0gem9uZVN5bWJvbCQxKCdPcmlnaW5hbERlbGVnYXRlJyk7XG4gICAgICAgIHZhciBQUk9NSVNFX1NZTUJPTCA9IHpvbmVTeW1ib2wkMSgnUHJvbWlzZScpO1xuICAgICAgICB2YXIgRVJST1JfU1lNQk9MID0gem9uZVN5bWJvbCQxKCdFcnJvcicpO1xuICAgICAgICB2YXIgbmV3RnVuY3Rpb25Ub1N0cmluZyA9IGZ1bmN0aW9uIHRvU3RyaW5nKCkge1xuICAgICAgICAgICAgaWYgKHR5cGVvZiB0aGlzID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgdmFyIG9yaWdpbmFsRGVsZWdhdGUgPSB0aGlzW09SSUdJTkFMX0RFTEVHQVRFX1NZTUJPTF07XG4gICAgICAgICAgICAgICAgaWYgKG9yaWdpbmFsRGVsZWdhdGUpIHtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHR5cGVvZiBvcmlnaW5hbERlbGVnYXRlID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gb3JpZ2luYWxGdW5jdGlvblRvU3RyaW5nLmNhbGwob3JpZ2luYWxEZWxlZ2F0ZSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gT2JqZWN0LnByb3RvdHlwZS50b1N0cmluZy5jYWxsKG9yaWdpbmFsRGVsZWdhdGUpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICh0aGlzID09PSBQcm9taXNlKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciBuYXRpdmVQcm9taXNlID0gZ2xvYmFsW1BST01JU0VfU1lNQk9MXTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5hdGl2ZVByb21pc2UpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBvcmlnaW5hbEZ1bmN0aW9uVG9TdHJpbmcuY2FsbChuYXRpdmVQcm9taXNlKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBpZiAodGhpcyA9PT0gRXJyb3IpIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5hdGl2ZUVycm9yID0gZ2xvYmFsW0VSUk9SX1NZTUJPTF07XG4gICAgICAgICAgICAgICAgICAgIGlmIChuYXRpdmVFcnJvcikge1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIG9yaWdpbmFsRnVuY3Rpb25Ub1N0cmluZy5jYWxsKG5hdGl2ZUVycm9yKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBvcmlnaW5hbEZ1bmN0aW9uVG9TdHJpbmcuY2FsbCh0aGlzKTtcbiAgICAgICAgfTtcbiAgICAgICAgbmV3RnVuY3Rpb25Ub1N0cmluZ1tPUklHSU5BTF9ERUxFR0FURV9TWU1CT0xdID0gb3JpZ2luYWxGdW5jdGlvblRvU3RyaW5nO1xuICAgICAgICBGdW5jdGlvbi5wcm90b3R5cGUudG9TdHJpbmcgPSBuZXdGdW5jdGlvblRvU3RyaW5nO1xuICAgICAgICAvLyBwYXRjaCBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nIHRvIGxldCB0aGVtIGxvb2sgbGlrZSBuYXRpdmVcbiAgICAgICAgdmFyIG9yaWdpbmFsT2JqZWN0VG9TdHJpbmcgPSBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nO1xuICAgICAgICB2YXIgUFJPTUlTRV9PQkpFQ1RfVE9fU1RSSU5HID0gJ1tvYmplY3QgUHJvbWlzZV0nO1xuICAgICAgICBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgaWYgKHR5cGVvZiBQcm9taXNlID09PSAnZnVuY3Rpb24nICYmIHRoaXMgaW5zdGFuY2VvZiBQcm9taXNlKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIFBST01JU0VfT0JKRUNUX1RPX1NUUklORztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBvcmlnaW5hbE9iamVjdFRvU3RyaW5nLmNhbGwodGhpcyk7XG4gICAgICAgIH07XG4gICAgfSk7XG4gICAgLyoqXG4gICAgICogQGxpY2Vuc2VcbiAgICAgKiBDb3B5cmlnaHQgR29vZ2xlIExMQyBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICAgICAqXG4gICAgICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAgICAgKiBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIGF0IGh0dHBzOi8vYW5ndWxhci5pby9saWNlbnNlXG4gICAgICovXG4gICAgdmFyIHBhc3NpdmVTdXBwb3J0ZWQgPSBmYWxzZTtcbiAgICBpZiAodHlwZW9mIHdpbmRvdyAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgIHZhciBvcHRpb25zID0gT2JqZWN0LmRlZmluZVByb3BlcnR5KHt9LCAncGFzc2l2ZScsIHtcbiAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgcGFzc2l2ZVN1cHBvcnRlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAvLyBOb3RlOiBXZSBwYXNzIHRoZSBgb3B0aW9uc2Agb2JqZWN0IGFzIHRoZSBldmVudCBoYW5kbGVyIHRvby4gVGhpcyBpcyBub3QgY29tcGF0aWJsZSB3aXRoIHRoZVxuICAgICAgICAgICAgLy8gc2lnbmF0dXJlIG9mIGBhZGRFdmVudExpc3RlbmVyYCBvciBgcmVtb3ZlRXZlbnRMaXN0ZW5lcmAgYnV0IGVuYWJsZXMgdXMgdG8gcmVtb3ZlIHRoZSBoYW5kbGVyXG4gICAgICAgICAgICAvLyB3aXRob3V0IGFuIGFjdHVhbCBoYW5kbGVyLlxuICAgICAgICAgICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ3Rlc3QnLCBvcHRpb25zLCBvcHRpb25zKTtcbiAgICAgICAgICAgIHdpbmRvdy5yZW1vdmVFdmVudExpc3RlbmVyKCd0ZXN0Jywgb3B0aW9ucywgb3B0aW9ucyk7XG4gICAgICAgIH1cbiAgICAgICAgY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgcGFzc2l2ZVN1cHBvcnRlZCA9IGZhbHNlO1xuICAgICAgICB9XG4gICAgfVxuICAgIC8vIGFuIGlkZW50aWZpZXIgdG8gdGVsbCBab25lVGFzayBkbyBub3QgY3JlYXRlIGEgbmV3IGludm9rZSBjbG9zdXJlXG4gICAgdmFyIE9QVElNSVpFRF9aT05FX0VWRU5UX1RBU0tfREFUQSA9IHtcbiAgICAgICAgdXNlRzogdHJ1ZVxuICAgIH07XG4gICAgdmFyIHpvbmVTeW1ib2xFdmVudE5hbWVzID0ge307XG4gICAgdmFyIGdsb2JhbFNvdXJjZXMgPSB7fTtcbiAgICB2YXIgRVZFTlRfTkFNRV9TWU1CT0xfUkVHWCA9IG5ldyBSZWdFeHAoJ14nICsgWk9ORV9TWU1CT0xfUFJFRklYICsgJyhcXFxcdyspKHRydWV8ZmFsc2UpJCcpO1xuICAgIHZhciBJTU1FRElBVEVfUFJPUEFHQVRJT05fU1lNQk9MID0gem9uZVN5bWJvbCQxKCdwcm9wYWdhdGlvblN0b3BwZWQnKTtcbiAgICBmdW5jdGlvbiBwcmVwYXJlRXZlbnROYW1lcyhldmVudE5hbWUsIGV2ZW50TmFtZVRvU3RyaW5nKSB7XG4gICAgICAgIHZhciBmYWxzZUV2ZW50TmFtZSA9IChldmVudE5hbWVUb1N0cmluZyA/IGV2ZW50TmFtZVRvU3RyaW5nKGV2ZW50TmFtZSkgOiBldmVudE5hbWUpICsgRkFMU0VfU1RSO1xuICAgICAgICB2YXIgdHJ1ZUV2ZW50TmFtZSA9IChldmVudE5hbWVUb1N0cmluZyA/IGV2ZW50TmFtZVRvU3RyaW5nKGV2ZW50TmFtZSkgOiBldmVudE5hbWUpICsgVFJVRV9TVFI7XG4gICAgICAgIHZhciBzeW1ib2wgPSBaT05FX1NZTUJPTF9QUkVGSVggKyBmYWxzZUV2ZW50TmFtZTtcbiAgICAgICAgdmFyIHN5bWJvbENhcHR1cmUgPSBaT05FX1NZTUJPTF9QUkVGSVggKyB0cnVlRXZlbnROYW1lO1xuICAgICAgICB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdID0ge307XG4gICAgICAgIHpvbmVTeW1ib2xFdmVudE5hbWVzW2V2ZW50TmFtZV1bRkFMU0VfU1RSXSA9IHN5bWJvbDtcbiAgICAgICAgem9uZVN5bWJvbEV2ZW50TmFtZXNbZXZlbnROYW1lXVtUUlVFX1NUUl0gPSBzeW1ib2xDYXB0dXJlO1xuICAgIH1cbiAgICBmdW5jdGlvbiBwYXRjaEV2ZW50VGFyZ2V0KF9nbG9iYWwsIGFwaSwgYXBpcywgcGF0Y2hPcHRpb25zKSB7XG4gICAgICAgIHZhciBBRERfRVZFTlRfTElTVEVORVIgPSAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy5hZGQpIHx8IEFERF9FVkVOVF9MSVNURU5FUl9TVFI7XG4gICAgICAgIHZhciBSRU1PVkVfRVZFTlRfTElTVEVORVIgPSAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy5ybSkgfHwgUkVNT1ZFX0VWRU5UX0xJU1RFTkVSX1NUUjtcbiAgICAgICAgdmFyIExJU1RFTkVSU19FVkVOVF9MSVNURU5FUiA9IChwYXRjaE9wdGlvbnMgJiYgcGF0Y2hPcHRpb25zLmxpc3RlbmVycykgfHwgJ2V2ZW50TGlzdGVuZXJzJztcbiAgICAgICAgdmFyIFJFTU9WRV9BTExfTElTVEVORVJTX0VWRU5UX0xJU1RFTkVSID0gKHBhdGNoT3B0aW9ucyAmJiBwYXRjaE9wdGlvbnMucm1BbGwpIHx8ICdyZW1vdmVBbGxMaXN0ZW5lcnMnO1xuICAgICAgICB2YXIgem9uZVN5bWJvbEFkZEV2ZW50TGlzdGVuZXIgPSB6b25lU3ltYm9sJDEoQUREX0VWRU5UX0xJU1RFTkVSKTtcbiAgICAgICAgdmFyIEFERF9FVkVOVF9MSVNURU5FUl9TT1VSQ0UgPSAnLicgKyBBRERfRVZFTlRfTElTVEVORVIgKyAnOic7XG4gICAgICAgIHZhciBQUkVQRU5EX0VWRU5UX0xJU1RFTkVSID0gJ3ByZXBlbmRMaXN0ZW5lcic7XG4gICAgICAgIHZhciBQUkVQRU5EX0VWRU5UX0xJU1RFTkVSX1NPVVJDRSA9ICcuJyArIFBSRVBFTkRfRVZFTlRfTElTVEVORVIgKyAnOic7XG4gICAgICAgIHZhciBpbnZva2VUYXNrID0gZnVuY3Rpb24gKHRhc2ssIHRhcmdldCwgZXZlbnQpIHtcbiAgICAgICAgICAgIC8vIGZvciBiZXR0ZXIgcGVyZm9ybWFuY2UsIGNoZWNrIGlzUmVtb3ZlZCB3aGljaCBpcyBzZXRcbiAgICAgICAgICAgIC8vIGJ5IHJlbW92ZUV2ZW50TGlzdGVuZXJcbiAgICAgICAgICAgIGlmICh0YXNrLmlzUmVtb3ZlZCkge1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBkZWxlZ2F0ZSA9IHRhc2suY2FsbGJhY2s7XG4gICAgICAgICAgICBpZiAodHlwZW9mIGRlbGVnYXRlID09PSAnb2JqZWN0JyAmJiBkZWxlZ2F0ZS5oYW5kbGVFdmVudCkge1xuICAgICAgICAgICAgICAgIC8vIGNyZWF0ZSB0aGUgYmluZCB2ZXJzaW9uIG9mIGhhbmRsZUV2ZW50IHdoZW4gaW52b2tlXG4gICAgICAgICAgICAgICAgdGFzay5jYWxsYmFjayA9IGZ1bmN0aW9uIChldmVudCkgeyByZXR1cm4gZGVsZWdhdGUuaGFuZGxlRXZlbnQoZXZlbnQpOyB9O1xuICAgICAgICAgICAgICAgIHRhc2sub3JpZ2luYWxEZWxlZ2F0ZSA9IGRlbGVnYXRlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgLy8gaW52b2tlIHN0YXRpYyB0YXNrLmludm9rZVxuICAgICAgICAgICAgLy8gbmVlZCB0byB0cnkvY2F0Y2ggZXJyb3IgaGVyZSwgb3RoZXJ3aXNlLCB0aGUgZXJyb3IgaW4gb25lIGV2ZW50IGxpc3RlbmVyXG4gICAgICAgICAgICAvLyB3aWxsIGJyZWFrIHRoZSBleGVjdXRpb25zIG9mIHRoZSBvdGhlciBldmVudCBsaXN0ZW5lcnMuIEFsc28gZXJyb3Igd2lsbFxuICAgICAgICAgICAgLy8gbm90IHJlbW92ZSB0aGUgZXZlbnQgbGlzdGVuZXIgd2hlbiBgb25jZWAgb3B0aW9ucyBpcyB0cnVlLlxuICAgICAgICAgICAgdmFyIGVycm9yO1xuICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICB0YXNrLmludm9rZSh0YXNrLCB0YXJnZXQsIFtldmVudF0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgICAgIGVycm9yID0gZXJyO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIG9wdGlvbnMgPSB0YXNrLm9wdGlvbnM7XG4gICAgICAgICAgICBpZiAob3B0aW9ucyAmJiB0eXBlb2Ygb3B0aW9ucyA9PT0gJ29iamVjdCcgJiYgb3B0aW9ucy5vbmNlKSB7XG4gICAgICAgICAgICAgICAgLy8gaWYgb3B0aW9ucy5vbmNlIGlzIHRydWUsIGFmdGVyIGludm9rZSBvbmNlIHJlbW92ZSBsaXN0ZW5lciBoZXJlXG4gICAgICAgICAgICAgICAgLy8gb25seSBicm93c2VyIG5lZWQgdG8gZG8gdGhpcywgbm9kZWpzIGV2ZW50RW1pdHRlciB3aWxsIGNhbCByZW1vdmVMaXN0ZW5lclxuICAgICAgICAgICAgICAgIC8vIGluc2lkZSBFdmVudEVtaXR0ZXIub25jZVxuICAgICAgICAgICAgICAgIHZhciBkZWxlZ2F0ZV8xID0gdGFzay5vcmlnaW5hbERlbGVnYXRlID8gdGFzay5vcmlnaW5hbERlbGVnYXRlIDogdGFzay5jYWxsYmFjaztcbiAgICAgICAgICAgICAgICB0YXJnZXRbUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXS5jYWxsKHRhcmdldCwgZXZlbnQudHlwZSwgZGVsZWdhdGVfMSwgb3B0aW9ucyk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gZXJyb3I7XG4gICAgICAgIH07XG4gICAgICAgIGZ1bmN0aW9uIGdsb2JhbENhbGxiYWNrKGNvbnRleHQsIGV2ZW50LCBpc0NhcHR1cmUpIHtcbiAgICAgICAgICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hbmd1bGFyL3pvbmUuanMvaXNzdWVzLzkxMSwgaW4gSUUsIHNvbWV0aW1lc1xuICAgICAgICAgICAgLy8gZXZlbnQgd2lsbCBiZSB1bmRlZmluZWQsIHNvIHdlIG5lZWQgdG8gdXNlIHdpbmRvdy5ldmVudFxuICAgICAgICAgICAgZXZlbnQgPSBldmVudCB8fCBfZ2xvYmFsLmV2ZW50O1xuICAgICAgICAgICAgaWYgKCFldmVudCkge1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8vIGV2ZW50LnRhcmdldCBpcyBuZWVkZWQgZm9yIFNhbXN1bmcgVFYgYW5kIFNvdXJjZUJ1ZmZlclxuICAgICAgICAgICAgLy8gfHwgZ2xvYmFsIGlzIG5lZWRlZCBodHRwczovL2dpdGh1Yi5jb20vYW5ndWxhci96b25lLmpzL2lzc3Vlcy8xOTBcbiAgICAgICAgICAgIHZhciB0YXJnZXQgPSBjb250ZXh0IHx8IGV2ZW50LnRhcmdldCB8fCBfZ2xvYmFsO1xuICAgICAgICAgICAgdmFyIHRhc2tzID0gdGFyZ2V0W3pvbmVTeW1ib2xFdmVudE5hbWVzW2V2ZW50LnR5cGVdW2lzQ2FwdHVyZSA/IFRSVUVfU1RSIDogRkFMU0VfU1RSXV07XG4gICAgICAgICAgICBpZiAodGFza3MpIHtcbiAgICAgICAgICAgICAgICB2YXIgZXJyb3JzID0gW107XG4gICAgICAgICAgICAgICAgLy8gaW52b2tlIGFsbCB0YXNrcyB3aGljaCBhdHRhY2hlZCB0byBjdXJyZW50IHRhcmdldCB3aXRoIGdpdmVuIGV2ZW50LnR5cGUgYW5kIGNhcHR1cmUgPSBmYWxzZVxuICAgICAgICAgICAgICAgIC8vIGZvciBwZXJmb3JtYW5jZSBjb25jZXJuLCBpZiB0YXNrLmxlbmd0aCA9PT0gMSwganVzdCBpbnZva2VcbiAgICAgICAgICAgICAgICBpZiAodGFza3MubGVuZ3RoID09PSAxKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciBlcnIgPSBpbnZva2VUYXNrKHRhc2tzWzBdLCB0YXJnZXQsIGV2ZW50KTtcbiAgICAgICAgICAgICAgICAgICAgZXJyICYmIGVycm9ycy5wdXNoKGVycik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAvLyBodHRwczovL2dpdGh1Yi5jb20vYW5ndWxhci96b25lLmpzL2lzc3Vlcy84MzZcbiAgICAgICAgICAgICAgICAgICAgLy8gY29weSB0aGUgdGFza3MgYXJyYXkgYmVmb3JlIGludm9rZSwgdG8gYXZvaWRcbiAgICAgICAgICAgICAgICAgICAgLy8gdGhlIGNhbGxiYWNrIHdpbGwgcmVtb3ZlIGl0c2VsZiBvciBvdGhlciBsaXN0ZW5lclxuICAgICAgICAgICAgICAgICAgICB2YXIgY29weVRhc2tzID0gdGFza3Muc2xpY2UoKTtcbiAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBjb3B5VGFza3MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChldmVudCAmJiBldmVudFtJTU1FRElBVEVfUFJPUEFHQVRJT05fU1lNQk9MXSA9PT0gdHJ1ZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGVyciA9IGludm9rZVRhc2soY29weVRhc2tzW2ldLCB0YXJnZXQsIGV2ZW50KTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGVyciAmJiBlcnJvcnMucHVzaChlcnIpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIC8vIFNpbmNlIHRoZXJlIGlzIG9ubHkgb25lIGVycm9yLCB3ZSBkb24ndCBuZWVkIHRvIHNjaGVkdWxlIG1pY3JvVGFza1xuICAgICAgICAgICAgICAgIC8vIHRvIHRocm93IHRoZSBlcnJvci5cbiAgICAgICAgICAgICAgICBpZiAoZXJyb3JzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgICAgICAgICAgICAgICB0aHJvdyBlcnJvcnNbMF07XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICB2YXIgX2xvb3BfNCA9IGZ1bmN0aW9uIChpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgZXJyID0gZXJyb3JzW2ldO1xuICAgICAgICAgICAgICAgICAgICAgICAgYXBpLm5hdGl2ZVNjaGVkdWxlTWljcm9UYXNrKGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aHJvdyBlcnI7XG4gICAgICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBlcnJvcnMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIF9sb29wXzQoaSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgLy8gZ2xvYmFsIHNoYXJlZCB6b25lQXdhcmVDYWxsYmFjayB0byBoYW5kbGUgYWxsIGV2ZW50IGNhbGxiYWNrIHdpdGggY2FwdHVyZSA9IGZhbHNlXG4gICAgICAgIHZhciBnbG9iYWxab25lQXdhcmVDYWxsYmFjayA9IGZ1bmN0aW9uIChldmVudCkge1xuICAgICAgICAgICAgcmV0dXJuIGdsb2JhbENhbGxiYWNrKHRoaXMsIGV2ZW50LCBmYWxzZSk7XG4gICAgICAgIH07XG4gICAgICAgIC8vIGdsb2JhbCBzaGFyZWQgem9uZUF3YXJlQ2FsbGJhY2sgdG8gaGFuZGxlIGFsbCBldmVudCBjYWxsYmFjayB3aXRoIGNhcHR1cmUgPSB0cnVlXG4gICAgICAgIHZhciBnbG9iYWxab25lQXdhcmVDYXB0dXJlQ2FsbGJhY2sgPSBmdW5jdGlvbiAoZXZlbnQpIHtcbiAgICAgICAgICAgIHJldHVybiBnbG9iYWxDYWxsYmFjayh0aGlzLCBldmVudCwgdHJ1ZSk7XG4gICAgICAgIH07XG4gICAgICAgIGZ1bmN0aW9uIHBhdGNoRXZlbnRUYXJnZXRNZXRob2RzKG9iaiwgcGF0Y2hPcHRpb25zKSB7XG4gICAgICAgICAgICBpZiAoIW9iaikge1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciB1c2VHbG9iYWxDYWxsYmFjayA9IHRydWU7XG4gICAgICAgICAgICBpZiAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy51c2VHICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICB1c2VHbG9iYWxDYWxsYmFjayA9IHBhdGNoT3B0aW9ucy51c2VHO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIHZhbGlkYXRlSGFuZGxlciA9IHBhdGNoT3B0aW9ucyAmJiBwYXRjaE9wdGlvbnMudmg7XG4gICAgICAgICAgICB2YXIgY2hlY2tEdXBsaWNhdGUgPSB0cnVlO1xuICAgICAgICAgICAgaWYgKHBhdGNoT3B0aW9ucyAmJiBwYXRjaE9wdGlvbnMuY2hrRHVwICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICBjaGVja0R1cGxpY2F0ZSA9IHBhdGNoT3B0aW9ucy5jaGtEdXA7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgcmV0dXJuVGFyZ2V0ID0gZmFsc2U7XG4gICAgICAgICAgICBpZiAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy5ydCAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuVGFyZ2V0ID0gcGF0Y2hPcHRpb25zLnJ0O1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIHByb3RvID0gb2JqO1xuICAgICAgICAgICAgd2hpbGUgKHByb3RvICYmICFwcm90by5oYXNPd25Qcm9wZXJ0eShBRERfRVZFTlRfTElTVEVORVIpKSB7XG4gICAgICAgICAgICAgICAgcHJvdG8gPSBPYmplY3RHZXRQcm90b3R5cGVPZihwcm90byk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAoIXByb3RvICYmIG9ialtBRERfRVZFTlRfTElTVEVORVJdKSB7XG4gICAgICAgICAgICAgICAgLy8gc29tZWhvdyB3ZSBkaWQgbm90IGZpbmQgaXQsIGJ1dCB3ZSBjYW4gc2VlIGl0LiBUaGlzIGhhcHBlbnMgb24gSUUgZm9yIFdpbmRvdyBwcm9wZXJ0aWVzLlxuICAgICAgICAgICAgICAgIHByb3RvID0gb2JqO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKCFwcm90bykge1xuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmIChwcm90b1t6b25lU3ltYm9sQWRkRXZlbnRMaXN0ZW5lcl0pIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgZXZlbnROYW1lVG9TdHJpbmcgPSBwYXRjaE9wdGlvbnMgJiYgcGF0Y2hPcHRpb25zLmV2ZW50TmFtZVRvU3RyaW5nO1xuICAgICAgICAgICAgLy8gYSBzaGFyZWQgZ2xvYmFsIHRhc2tEYXRhIHRvIHBhc3MgZGF0YSBmb3Igc2NoZWR1bGVFdmVudFRhc2tcbiAgICAgICAgICAgIC8vIHNvIHdlIGRvIG5vdCBuZWVkIHRvIGNyZWF0ZSBhIG5ldyBvYmplY3QganVzdCBmb3IgcGFzcyBzb21lIGRhdGFcbiAgICAgICAgICAgIHZhciB0YXNrRGF0YSA9IHt9O1xuICAgICAgICAgICAgdmFyIG5hdGl2ZUFkZEV2ZW50TGlzdGVuZXIgPSBwcm90b1t6b25lU3ltYm9sQWRkRXZlbnRMaXN0ZW5lcl0gPSBwcm90b1tBRERfRVZFTlRfTElTVEVORVJdO1xuICAgICAgICAgICAgdmFyIG5hdGl2ZVJlbW92ZUV2ZW50TGlzdGVuZXIgPSBwcm90b1t6b25lU3ltYm9sJDEoUkVNT1ZFX0VWRU5UX0xJU1RFTkVSKV0gPVxuICAgICAgICAgICAgICAgIHByb3RvW1JFTU9WRV9FVkVOVF9MSVNURU5FUl07XG4gICAgICAgICAgICB2YXIgbmF0aXZlTGlzdGVuZXJzID0gcHJvdG9bem9uZVN5bWJvbCQxKExJU1RFTkVSU19FVkVOVF9MSVNURU5FUildID1cbiAgICAgICAgICAgICAgICBwcm90b1tMSVNURU5FUlNfRVZFTlRfTElTVEVORVJdO1xuICAgICAgICAgICAgdmFyIG5hdGl2ZVJlbW92ZUFsbExpc3RlbmVycyA9IHByb3RvW3pvbmVTeW1ib2wkMShSRU1PVkVfQUxMX0xJU1RFTkVSU19FVkVOVF9MSVNURU5FUildID1cbiAgICAgICAgICAgICAgICBwcm90b1tSRU1PVkVfQUxMX0xJU1RFTkVSU19FVkVOVF9MSVNURU5FUl07XG4gICAgICAgICAgICB2YXIgbmF0aXZlUHJlcGVuZEV2ZW50TGlzdGVuZXI7XG4gICAgICAgICAgICBpZiAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy5wcmVwZW5kKSB7XG4gICAgICAgICAgICAgICAgbmF0aXZlUHJlcGVuZEV2ZW50TGlzdGVuZXIgPSBwcm90b1t6b25lU3ltYm9sJDEocGF0Y2hPcHRpb25zLnByZXBlbmQpXSA9XG4gICAgICAgICAgICAgICAgICAgIHByb3RvW3BhdGNoT3B0aW9ucy5wcmVwZW5kXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8qKlxuICAgICAgICAgICAgICogVGhpcyB1dGlsIGZ1bmN0aW9uIHdpbGwgYnVpbGQgYW4gb3B0aW9uIG9iamVjdCB3aXRoIHBhc3NpdmUgb3B0aW9uXG4gICAgICAgICAgICAgKiB0byBoYW5kbGUgYWxsIHBvc3NpYmxlIGlucHV0IGZyb20gdGhlIHVzZXIuXG4gICAgICAgICAgICAgKi9cbiAgICAgICAgICAgIGZ1bmN0aW9uIGJ1aWxkRXZlbnRMaXN0ZW5lck9wdGlvbnMob3B0aW9ucywgcGFzc2l2ZSkge1xuICAgICAgICAgICAgICAgIGlmICghcGFzc2l2ZVN1cHBvcnRlZCAmJiB0eXBlb2Ygb3B0aW9ucyA9PT0gJ29iamVjdCcgJiYgb3B0aW9ucykge1xuICAgICAgICAgICAgICAgICAgICAvLyBkb2Vzbid0IHN1cHBvcnQgcGFzc2l2ZSBidXQgdXNlciB3YW50IHRvIHBhc3MgYW4gb2JqZWN0IGFzIG9wdGlvbnMuXG4gICAgICAgICAgICAgICAgICAgIC8vIHRoaXMgd2lsbCBub3Qgd29yayBvbiBzb21lIG9sZCBicm93c2VyLCBzbyB3ZSBqdXN0IHBhc3MgYSBib29sZWFuXG4gICAgICAgICAgICAgICAgICAgIC8vIGFzIHVzZUNhcHR1cmUgcGFyYW1ldGVyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAhIW9wdGlvbnMuY2FwdHVyZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKCFwYXNzaXZlU3VwcG9ydGVkIHx8ICFwYXNzaXZlKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBvcHRpb25zO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIG9wdGlvbnMgPT09ICdib29sZWFuJykge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4geyBjYXB0dXJlOiBvcHRpb25zLCBwYXNzaXZlOiB0cnVlIH07XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICghb3B0aW9ucykge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4geyBwYXNzaXZlOiB0cnVlIH07XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICh0eXBlb2Ygb3B0aW9ucyA9PT0gJ29iamVjdCcgJiYgb3B0aW9ucy5wYXNzaXZlICE9PSBmYWxzZSkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gT2JqZWN0LmFzc2lnbihPYmplY3QuYXNzaWduKHt9LCBvcHRpb25zKSwgeyBwYXNzaXZlOiB0cnVlIH0pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gb3B0aW9ucztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBjdXN0b21TY2hlZHVsZUdsb2JhbCA9IGZ1bmN0aW9uICh0YXNrKSB7XG4gICAgICAgICAgICAgICAgLy8gaWYgdGhlcmUgaXMgYWxyZWFkeSBhIHRhc2sgZm9yIHRoZSBldmVudE5hbWUgKyBjYXB0dXJlLFxuICAgICAgICAgICAgICAgIC8vIGp1c3QgcmV0dXJuLCBiZWNhdXNlIHdlIHVzZSB0aGUgc2hhcmVkIGdsb2JhbFpvbmVBd2FyZUNhbGxiYWNrIGhlcmUuXG4gICAgICAgICAgICAgICAgaWYgKHRhc2tEYXRhLmlzRXhpc3RpbmcpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gbmF0aXZlQWRkRXZlbnRMaXN0ZW5lci5jYWxsKHRhc2tEYXRhLnRhcmdldCwgdGFza0RhdGEuZXZlbnROYW1lLCB0YXNrRGF0YS5jYXB0dXJlID8gZ2xvYmFsWm9uZUF3YXJlQ2FwdHVyZUNhbGxiYWNrIDogZ2xvYmFsWm9uZUF3YXJlQ2FsbGJhY2ssIHRhc2tEYXRhLm9wdGlvbnMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHZhciBjdXN0b21DYW5jZWxHbG9iYWwgPSBmdW5jdGlvbiAodGFzaykge1xuICAgICAgICAgICAgICAgIC8vIGlmIHRhc2sgaXMgbm90IG1hcmtlZCBhcyBpc1JlbW92ZWQsIHRoaXMgY2FsbCBpcyBkaXJlY3RseVxuICAgICAgICAgICAgICAgIC8vIGZyb20gWm9uZS5wcm90b3R5cGUuY2FuY2VsVGFzaywgd2Ugc2hvdWxkIHJlbW92ZSB0aGUgdGFza1xuICAgICAgICAgICAgICAgIC8vIGZyb20gdGFza3NMaXN0IG9mIHRhcmdldCBmaXJzdFxuICAgICAgICAgICAgICAgIGlmICghdGFzay5pc1JlbW92ZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHN5bWJvbEV2ZW50TmFtZXMgPSB6b25lU3ltYm9sRXZlbnROYW1lc1t0YXNrLmV2ZW50TmFtZV07XG4gICAgICAgICAgICAgICAgICAgIHZhciBzeW1ib2xFdmVudE5hbWUgPSB2b2lkIDA7XG4gICAgICAgICAgICAgICAgICAgIGlmIChzeW1ib2xFdmVudE5hbWVzKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBzeW1ib2xFdmVudE5hbWUgPSBzeW1ib2xFdmVudE5hbWVzW3Rhc2suY2FwdHVyZSA/IFRSVUVfU1RSIDogRkFMU0VfU1RSXTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YXIgZXhpc3RpbmdUYXNrcyA9IHN5bWJvbEV2ZW50TmFtZSAmJiB0YXNrLnRhcmdldFtzeW1ib2xFdmVudE5hbWVdO1xuICAgICAgICAgICAgICAgICAgICBpZiAoZXhpc3RpbmdUYXNrcykge1xuICAgICAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBleGlzdGluZ1Rhc2tzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGV4aXN0aW5nVGFzayA9IGV4aXN0aW5nVGFza3NbaV07XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGV4aXN0aW5nVGFzayA9PT0gdGFzaykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBleGlzdGluZ1Rhc2tzLnNwbGljZShpLCAxKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gc2V0IGlzUmVtb3ZlZCB0byBkYXRhIGZvciBmYXN0ZXIgaW52b2tlVGFzayBjaGVja1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YXNrLmlzUmVtb3ZlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChleGlzdGluZ1Rhc2tzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxsIHRhc2tzIGZvciB0aGUgZXZlbnROYW1lICsgY2FwdHVyZSBoYXZlIGdvbmUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyByZW1vdmUgZ2xvYmFsWm9uZUF3YXJlQ2FsbGJhY2sgYW5kIHJlbW92ZSB0aGUgdGFzayBjYWNoZSBmcm9tIHRhcmdldFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGFzay5hbGxSZW1vdmVkID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhc2sudGFyZ2V0W3N5bWJvbEV2ZW50TmFtZV0gPSBudWxsO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAvLyBpZiBhbGwgdGFza3MgZm9yIHRoZSBldmVudE5hbWUgKyBjYXB0dXJlIGhhdmUgZ29uZSxcbiAgICAgICAgICAgICAgICAvLyB3ZSB3aWxsIHJlYWxseSByZW1vdmUgdGhlIGdsb2JhbCBldmVudCBjYWxsYmFjayxcbiAgICAgICAgICAgICAgICAvLyBpZiBub3QsIHJldHVyblxuICAgICAgICAgICAgICAgIGlmICghdGFzay5hbGxSZW1vdmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZVJlbW92ZUV2ZW50TGlzdGVuZXIuY2FsbCh0YXNrLnRhcmdldCwgdGFzay5ldmVudE5hbWUsIHRhc2suY2FwdHVyZSA/IGdsb2JhbFpvbmVBd2FyZUNhcHR1cmVDYWxsYmFjayA6IGdsb2JhbFpvbmVBd2FyZUNhbGxiYWNrLCB0YXNrLm9wdGlvbnMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHZhciBjdXN0b21TY2hlZHVsZU5vbkdsb2JhbCA9IGZ1bmN0aW9uICh0YXNrKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZUFkZEV2ZW50TGlzdGVuZXIuY2FsbCh0YXNrRGF0YS50YXJnZXQsIHRhc2tEYXRhLmV2ZW50TmFtZSwgdGFzay5pbnZva2UsIHRhc2tEYXRhLm9wdGlvbnMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHZhciBjdXN0b21TY2hlZHVsZVByZXBlbmQgPSBmdW5jdGlvbiAodGFzaykge1xuICAgICAgICAgICAgICAgIHJldHVybiBuYXRpdmVQcmVwZW5kRXZlbnRMaXN0ZW5lci5jYWxsKHRhc2tEYXRhLnRhcmdldCwgdGFza0RhdGEuZXZlbnROYW1lLCB0YXNrLmludm9rZSwgdGFza0RhdGEub3B0aW9ucyk7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgdmFyIGN1c3RvbUNhbmNlbE5vbkdsb2JhbCA9IGZ1bmN0aW9uICh0YXNrKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZVJlbW92ZUV2ZW50TGlzdGVuZXIuY2FsbCh0YXNrLnRhcmdldCwgdGFzay5ldmVudE5hbWUsIHRhc2suaW52b2tlLCB0YXNrLm9wdGlvbnMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHZhciBjdXN0b21TY2hlZHVsZSA9IHVzZUdsb2JhbENhbGxiYWNrID8gY3VzdG9tU2NoZWR1bGVHbG9iYWwgOiBjdXN0b21TY2hlZHVsZU5vbkdsb2JhbDtcbiAgICAgICAgICAgIHZhciBjdXN0b21DYW5jZWwgPSB1c2VHbG9iYWxDYWxsYmFjayA/IGN1c3RvbUNhbmNlbEdsb2JhbCA6IGN1c3RvbUNhbmNlbE5vbkdsb2JhbDtcbiAgICAgICAgICAgIHZhciBjb21wYXJlVGFza0NhbGxiYWNrVnNEZWxlZ2F0ZSA9IGZ1bmN0aW9uICh0YXNrLCBkZWxlZ2F0ZSkge1xuICAgICAgICAgICAgICAgIHZhciB0eXBlT2ZEZWxlZ2F0ZSA9IHR5cGVvZiBkZWxlZ2F0ZTtcbiAgICAgICAgICAgICAgICByZXR1cm4gKHR5cGVPZkRlbGVnYXRlID09PSAnZnVuY3Rpb24nICYmIHRhc2suY2FsbGJhY2sgPT09IGRlbGVnYXRlKSB8fFxuICAgICAgICAgICAgICAgICAgICAodHlwZU9mRGVsZWdhdGUgPT09ICdvYmplY3QnICYmIHRhc2sub3JpZ2luYWxEZWxlZ2F0ZSA9PT0gZGVsZWdhdGUpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHZhciBjb21wYXJlID0gKHBhdGNoT3B0aW9ucyAmJiBwYXRjaE9wdGlvbnMuZGlmZikgPyBwYXRjaE9wdGlvbnMuZGlmZiA6IGNvbXBhcmVUYXNrQ2FsbGJhY2tWc0RlbGVnYXRlO1xuICAgICAgICAgICAgdmFyIHVucGF0Y2hlZEV2ZW50cyA9IFpvbmVbem9uZVN5bWJvbCQxKCdVTlBBVENIRURfRVZFTlRTJyldO1xuICAgICAgICAgICAgdmFyIHBhc3NpdmVFdmVudHMgPSBfZ2xvYmFsW3pvbmVTeW1ib2wkMSgnUEFTU0lWRV9FVkVOVFMnKV07XG4gICAgICAgICAgICB2YXIgbWFrZUFkZExpc3RlbmVyID0gZnVuY3Rpb24gKG5hdGl2ZUxpc3RlbmVyLCBhZGRTb3VyY2UsIGN1c3RvbVNjaGVkdWxlRm4sIGN1c3RvbUNhbmNlbEZuLCByZXR1cm5UYXJnZXQsIHByZXBlbmQpIHtcbiAgICAgICAgICAgICAgICBpZiAocmV0dXJuVGFyZ2V0ID09PSB2b2lkIDApIHsgcmV0dXJuVGFyZ2V0ID0gZmFsc2U7IH1cbiAgICAgICAgICAgICAgICBpZiAocHJlcGVuZCA9PT0gdm9pZCAwKSB7IHByZXBlbmQgPSBmYWxzZTsgfVxuICAgICAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciB0YXJnZXQgPSB0aGlzIHx8IF9nbG9iYWw7XG4gICAgICAgICAgICAgICAgICAgIHZhciBldmVudE5hbWUgPSBhcmd1bWVudHNbMF07XG4gICAgICAgICAgICAgICAgICAgIGlmIChwYXRjaE9wdGlvbnMgJiYgcGF0Y2hPcHRpb25zLnRyYW5zZmVyRXZlbnROYW1lKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBldmVudE5hbWUgPSBwYXRjaE9wdGlvbnMudHJhbnNmZXJFdmVudE5hbWUoZXZlbnROYW1lKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YXIgZGVsZWdhdGUgPSBhcmd1bWVudHNbMV07XG4gICAgICAgICAgICAgICAgICAgIGlmICghZGVsZWdhdGUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBuYXRpdmVMaXN0ZW5lci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGlmIChpc05vZGUgJiYgZXZlbnROYW1lID09PSAndW5jYXVnaHRFeGNlcHRpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBkb24ndCBwYXRjaCB1bmNhdWdodEV4Y2VwdGlvbiBvZiBub2RlanMgdG8gcHJldmVudCBlbmRsZXNzIGxvb3BcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBuYXRpdmVMaXN0ZW5lci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIC8vIGRvbid0IGNyZWF0ZSB0aGUgYmluZCBkZWxlZ2F0ZSBmdW5jdGlvbiBmb3IgaGFuZGxlRXZlbnRcbiAgICAgICAgICAgICAgICAgICAgLy8gY2FzZSBoZXJlIHRvIGltcHJvdmUgYWRkRXZlbnRMaXN0ZW5lciBwZXJmb3JtYW5jZVxuICAgICAgICAgICAgICAgICAgICAvLyB3ZSB3aWxsIGNyZWF0ZSB0aGUgYmluZCBkZWxlZ2F0ZSB3aGVuIGludm9rZVxuICAgICAgICAgICAgICAgICAgICB2YXIgaXNIYW5kbGVFdmVudCA9IGZhbHNlO1xuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGRlbGVnYXRlICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoIWRlbGVnYXRlLmhhbmRsZUV2ZW50KSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZUxpc3RlbmVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICBpc0hhbmRsZUV2ZW50ID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAodmFsaWRhdGVIYW5kbGVyICYmICF2YWxpZGF0ZUhhbmRsZXIobmF0aXZlTGlzdGVuZXIsIGRlbGVnYXRlLCB0YXJnZXQsIGFyZ3VtZW50cykpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YXIgcGFzc2l2ZSA9IHBhc3NpdmVTdXBwb3J0ZWQgJiYgISFwYXNzaXZlRXZlbnRzICYmIHBhc3NpdmVFdmVudHMuaW5kZXhPZihldmVudE5hbWUpICE9PSAtMTtcbiAgICAgICAgICAgICAgICAgICAgdmFyIG9wdGlvbnMgPSBidWlsZEV2ZW50TGlzdGVuZXJPcHRpb25zKGFyZ3VtZW50c1syXSwgcGFzc2l2ZSk7XG4gICAgICAgICAgICAgICAgICAgIGlmICh1bnBhdGNoZWRFdmVudHMpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGNoZWNrIHVucGF0Y2hlZCBsaXN0XG4gICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHVucGF0Y2hlZEV2ZW50cy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChldmVudE5hbWUgPT09IHVucGF0Y2hlZEV2ZW50c1tpXSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAocGFzc2l2ZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZUxpc3RlbmVyLmNhbGwodGFyZ2V0LCBldmVudE5hbWUsIGRlbGVnYXRlLCBvcHRpb25zKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBuYXRpdmVMaXN0ZW5lci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHZhciBjYXB0dXJlID0gIW9wdGlvbnMgPyBmYWxzZSA6IHR5cGVvZiBvcHRpb25zID09PSAnYm9vbGVhbicgPyB0cnVlIDogb3B0aW9ucy5jYXB0dXJlO1xuICAgICAgICAgICAgICAgICAgICB2YXIgb25jZSA9IG9wdGlvbnMgJiYgdHlwZW9mIG9wdGlvbnMgPT09ICdvYmplY3QnID8gb3B0aW9ucy5vbmNlIDogZmFsc2U7XG4gICAgICAgICAgICAgICAgICAgIHZhciB6b25lID0gWm9uZS5jdXJyZW50O1xuICAgICAgICAgICAgICAgICAgICB2YXIgc3ltYm9sRXZlbnROYW1lcyA9IHpvbmVTeW1ib2xFdmVudE5hbWVzW2V2ZW50TmFtZV07XG4gICAgICAgICAgICAgICAgICAgIGlmICghc3ltYm9sRXZlbnROYW1lcykge1xuICAgICAgICAgICAgICAgICAgICAgICAgcHJlcGFyZUV2ZW50TmFtZXMoZXZlbnROYW1lLCBldmVudE5hbWVUb1N0cmluZyk7XG4gICAgICAgICAgICAgICAgICAgICAgICBzeW1ib2xFdmVudE5hbWVzID0gem9uZVN5bWJvbEV2ZW50TmFtZXNbZXZlbnROYW1lXTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB2YXIgc3ltYm9sRXZlbnROYW1lID0gc3ltYm9sRXZlbnROYW1lc1tjYXB0dXJlID8gVFJVRV9TVFIgOiBGQUxTRV9TVFJdO1xuICAgICAgICAgICAgICAgICAgICB2YXIgZXhpc3RpbmdUYXNrcyA9IHRhcmdldFtzeW1ib2xFdmVudE5hbWVdO1xuICAgICAgICAgICAgICAgICAgICB2YXIgaXNFeGlzdGluZyA9IGZhbHNlO1xuICAgICAgICAgICAgICAgICAgICBpZiAoZXhpc3RpbmdUYXNrcykge1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gYWxyZWFkeSBoYXZlIHRhc2sgcmVnaXN0ZXJlZFxuICAgICAgICAgICAgICAgICAgICAgICAgaXNFeGlzdGluZyA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoY2hlY2tEdXBsaWNhdGUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGV4aXN0aW5nVGFza3MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGNvbXBhcmUoZXhpc3RpbmdUYXNrc1tpXSwgZGVsZWdhdGUpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBzYW1lIGNhbGxiYWNrLCBzYW1lIGNhcHR1cmUsIHNhbWUgZXZlbnQgbmFtZSwganVzdCByZXR1cm5cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGV4aXN0aW5nVGFza3MgPSB0YXJnZXRbc3ltYm9sRXZlbnROYW1lXSA9IFtdO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHZhciBzb3VyY2U7XG4gICAgICAgICAgICAgICAgICAgIHZhciBjb25zdHJ1Y3Rvck5hbWUgPSB0YXJnZXQuY29uc3RydWN0b3JbJ25hbWUnXTtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHRhcmdldFNvdXJjZSA9IGdsb2JhbFNvdXJjZXNbY29uc3RydWN0b3JOYW1lXTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRhcmdldFNvdXJjZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgc291cmNlID0gdGFyZ2V0U291cmNlW2V2ZW50TmFtZV07XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgaWYgKCFzb3VyY2UpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZSA9IGNvbnN0cnVjdG9yTmFtZSArIGFkZFNvdXJjZSArXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgKGV2ZW50TmFtZVRvU3RyaW5nID8gZXZlbnROYW1lVG9TdHJpbmcoZXZlbnROYW1lKSA6IGV2ZW50TmFtZSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgLy8gZG8gbm90IGNyZWF0ZSBhIG5ldyBvYmplY3QgYXMgdGFzay5kYXRhIHRvIHBhc3MgdGhvc2UgdGhpbmdzXG4gICAgICAgICAgICAgICAgICAgIC8vIGp1c3QgdXNlIHRoZSBnbG9iYWwgc2hhcmVkIG9uZVxuICAgICAgICAgICAgICAgICAgICB0YXNrRGF0YS5vcHRpb25zID0gb3B0aW9ucztcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9uY2UpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGlmIGFkZEV2ZW50TGlzdGVuZXIgd2l0aCBvbmNlIG9wdGlvbnMsIHdlIGRvbid0IHBhc3MgaXQgdG9cbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIG5hdGl2ZSBhZGRFdmVudExpc3RlbmVyLCBpbnN0ZWFkIHdlIGtlZXAgdGhlIG9uY2Ugc2V0dGluZ1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gYW5kIGhhbmRsZSBvdXJzZWx2ZXMuXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXNrRGF0YS5vcHRpb25zLm9uY2UgPSBmYWxzZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB0YXNrRGF0YS50YXJnZXQgPSB0YXJnZXQ7XG4gICAgICAgICAgICAgICAgICAgIHRhc2tEYXRhLmNhcHR1cmUgPSBjYXB0dXJlO1xuICAgICAgICAgICAgICAgICAgICB0YXNrRGF0YS5ldmVudE5hbWUgPSBldmVudE5hbWU7XG4gICAgICAgICAgICAgICAgICAgIHRhc2tEYXRhLmlzRXhpc3RpbmcgPSBpc0V4aXN0aW5nO1xuICAgICAgICAgICAgICAgICAgICB2YXIgZGF0YSA9IHVzZUdsb2JhbENhbGxiYWNrID8gT1BUSU1JWkVEX1pPTkVfRVZFTlRfVEFTS19EQVRBIDogdW5kZWZpbmVkO1xuICAgICAgICAgICAgICAgICAgICAvLyBrZWVwIHRhc2tEYXRhIGludG8gZGF0YSB0byBhbGxvdyBvblNjaGVkdWxlRXZlbnRUYXNrIHRvIGFjY2VzcyB0aGUgdGFzayBpbmZvcm1hdGlvblxuICAgICAgICAgICAgICAgICAgICBpZiAoZGF0YSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YS50YXNrRGF0YSA9IHRhc2tEYXRhO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHZhciB0YXNrID0gem9uZS5zY2hlZHVsZUV2ZW50VGFzayhzb3VyY2UsIGRlbGVnYXRlLCBkYXRhLCBjdXN0b21TY2hlZHVsZUZuLCBjdXN0b21DYW5jZWxGbik7XG4gICAgICAgICAgICAgICAgICAgIC8vIHNob3VsZCBjbGVhciB0YXNrRGF0YS50YXJnZXQgdG8gYXZvaWQgbWVtb3J5IGxlYWtcbiAgICAgICAgICAgICAgICAgICAgLy8gaXNzdWUsIGh0dHBzOi8vZ2l0aHViLmNvbS9hbmd1bGFyL2FuZ3VsYXIvaXNzdWVzLzIwNDQyXG4gICAgICAgICAgICAgICAgICAgIHRhc2tEYXRhLnRhcmdldCA9IG51bGw7XG4gICAgICAgICAgICAgICAgICAgIC8vIG5lZWQgdG8gY2xlYXIgdXAgdGFza0RhdGEgYmVjYXVzZSBpdCBpcyBhIGdsb2JhbCBvYmplY3RcbiAgICAgICAgICAgICAgICAgICAgaWYgKGRhdGEpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGEudGFza0RhdGEgPSBudWxsO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIC8vIGhhdmUgdG8gc2F2ZSB0aG9zZSBpbmZvcm1hdGlvbiB0byB0YXNrIGluIGNhc2VcbiAgICAgICAgICAgICAgICAgICAgLy8gYXBwbGljYXRpb24gbWF5IGNhbGwgdGFzay56b25lLmNhbmNlbFRhc2soKSBkaXJlY3RseVxuICAgICAgICAgICAgICAgICAgICBpZiAob25jZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgb3B0aW9ucy5vbmNlID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAoISghcGFzc2l2ZVN1cHBvcnRlZCAmJiB0eXBlb2YgdGFzay5vcHRpb25zID09PSAnYm9vbGVhbicpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBpZiBub3Qgc3VwcG9ydCBwYXNzaXZlLCBhbmQgd2UgcGFzcyBhbiBvcHRpb24gb2JqZWN0XG4gICAgICAgICAgICAgICAgICAgICAgICAvLyB0byBhZGRFdmVudExpc3RlbmVyLCB3ZSBzaG91bGQgc2F2ZSB0aGUgb3B0aW9ucyB0byB0YXNrXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXNrLm9wdGlvbnMgPSBvcHRpb25zO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHRhc2sudGFyZ2V0ID0gdGFyZ2V0O1xuICAgICAgICAgICAgICAgICAgICB0YXNrLmNhcHR1cmUgPSBjYXB0dXJlO1xuICAgICAgICAgICAgICAgICAgICB0YXNrLmV2ZW50TmFtZSA9IGV2ZW50TmFtZTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKGlzSGFuZGxlRXZlbnQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHNhdmUgb3JpZ2luYWwgZGVsZWdhdGUgZm9yIGNvbXBhcmUgdG8gY2hlY2sgZHVwbGljYXRlXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXNrLm9yaWdpbmFsRGVsZWdhdGUgPSBkZWxlZ2F0ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAoIXByZXBlbmQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGV4aXN0aW5nVGFza3MucHVzaCh0YXNrKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGV4aXN0aW5nVGFza3MudW5zaGlmdCh0YXNrKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAocmV0dXJuVGFyZ2V0KSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdGFyZ2V0O1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBwcm90b1tBRERfRVZFTlRfTElTVEVORVJdID0gbWFrZUFkZExpc3RlbmVyKG5hdGl2ZUFkZEV2ZW50TGlzdGVuZXIsIEFERF9FVkVOVF9MSVNURU5FUl9TT1VSQ0UsIGN1c3RvbVNjaGVkdWxlLCBjdXN0b21DYW5jZWwsIHJldHVyblRhcmdldCk7XG4gICAgICAgICAgICBpZiAobmF0aXZlUHJlcGVuZEV2ZW50TGlzdGVuZXIpIHtcbiAgICAgICAgICAgICAgICBwcm90b1tQUkVQRU5EX0VWRU5UX0xJU1RFTkVSXSA9IG1ha2VBZGRMaXN0ZW5lcihuYXRpdmVQcmVwZW5kRXZlbnRMaXN0ZW5lciwgUFJFUEVORF9FVkVOVF9MSVNURU5FUl9TT1VSQ0UsIGN1c3RvbVNjaGVkdWxlUHJlcGVuZCwgY3VzdG9tQ2FuY2VsLCByZXR1cm5UYXJnZXQsIHRydWUpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcHJvdG9bUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXSA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICB2YXIgdGFyZ2V0ID0gdGhpcyB8fCBfZ2xvYmFsO1xuICAgICAgICAgICAgICAgIHZhciBldmVudE5hbWUgPSBhcmd1bWVudHNbMF07XG4gICAgICAgICAgICAgICAgaWYgKHBhdGNoT3B0aW9ucyAmJiBwYXRjaE9wdGlvbnMudHJhbnNmZXJFdmVudE5hbWUpIHtcbiAgICAgICAgICAgICAgICAgICAgZXZlbnROYW1lID0gcGF0Y2hPcHRpb25zLnRyYW5zZmVyRXZlbnROYW1lKGV2ZW50TmFtZSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHZhciBvcHRpb25zID0gYXJndW1lbnRzWzJdO1xuICAgICAgICAgICAgICAgIHZhciBjYXB0dXJlID0gIW9wdGlvbnMgPyBmYWxzZSA6IHR5cGVvZiBvcHRpb25zID09PSAnYm9vbGVhbicgPyB0cnVlIDogb3B0aW9ucy5jYXB0dXJlO1xuICAgICAgICAgICAgICAgIHZhciBkZWxlZ2F0ZSA9IGFyZ3VtZW50c1sxXTtcbiAgICAgICAgICAgICAgICBpZiAoIWRlbGVnYXRlKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBuYXRpdmVSZW1vdmVFdmVudExpc3RlbmVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICh2YWxpZGF0ZUhhbmRsZXIgJiZcbiAgICAgICAgICAgICAgICAgICAgIXZhbGlkYXRlSGFuZGxlcihuYXRpdmVSZW1vdmVFdmVudExpc3RlbmVyLCBkZWxlZ2F0ZSwgdGFyZ2V0LCBhcmd1bWVudHMpKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdmFyIHN5bWJvbEV2ZW50TmFtZXMgPSB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdO1xuICAgICAgICAgICAgICAgIHZhciBzeW1ib2xFdmVudE5hbWU7XG4gICAgICAgICAgICAgICAgaWYgKHN5bWJvbEV2ZW50TmFtZXMpIHtcbiAgICAgICAgICAgICAgICAgICAgc3ltYm9sRXZlbnROYW1lID0gc3ltYm9sRXZlbnROYW1lc1tjYXB0dXJlID8gVFJVRV9TVFIgOiBGQUxTRV9TVFJdO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB2YXIgZXhpc3RpbmdUYXNrcyA9IHN5bWJvbEV2ZW50TmFtZSAmJiB0YXJnZXRbc3ltYm9sRXZlbnROYW1lXTtcbiAgICAgICAgICAgICAgICBpZiAoZXhpc3RpbmdUYXNrcykge1xuICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGV4aXN0aW5nVGFza3MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBleGlzdGluZ1Rhc2sgPSBleGlzdGluZ1Rhc2tzW2ldO1xuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGNvbXBhcmUoZXhpc3RpbmdUYXNrLCBkZWxlZ2F0ZSkpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBleGlzdGluZ1Rhc2tzLnNwbGljZShpLCAxKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBzZXQgaXNSZW1vdmVkIHRvIGRhdGEgZm9yIGZhc3RlciBpbnZva2VUYXNrIGNoZWNrXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZXhpc3RpbmdUYXNrLmlzUmVtb3ZlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGV4aXN0aW5nVGFza3MubGVuZ3RoID09PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGFsbCB0YXNrcyBmb3IgdGhlIGV2ZW50TmFtZSArIGNhcHR1cmUgaGF2ZSBnb25lLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyByZW1vdmUgZ2xvYmFsWm9uZUF3YXJlQ2FsbGJhY2sgYW5kIHJlbW92ZSB0aGUgdGFzayBjYWNoZSBmcm9tIHRhcmdldFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBleGlzdGluZ1Rhc2suYWxsUmVtb3ZlZCA9IHRydWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhcmdldFtzeW1ib2xFdmVudE5hbWVdID0gbnVsbDtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gaW4gdGhlIHRhcmdldCwgd2UgaGF2ZSBhbiBldmVudCBsaXN0ZW5lciB3aGljaCBpcyBhZGRlZCBieSBvbl9wcm9wZXJ0eVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBzdWNoIGFzIHRhcmdldC5vbmNsaWNrID0gZnVuY3Rpb24oKSB7fSwgc28gd2UgbmVlZCB0byBjbGVhciB0aGlzIGludGVybmFsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIHByb3BlcnR5IHRvbyBpZiBhbGwgZGVsZWdhdGVzIGFsbCByZW1vdmVkXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgZXZlbnROYW1lID09PSAnc3RyaW5nJykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIG9uUHJvcGVydHlTeW1ib2wgPSBaT05FX1NZTUJPTF9QUkVGSVggKyAnT05fUFJPUEVSVFknICsgZXZlbnROYW1lO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGFyZ2V0W29uUHJvcGVydHlTeW1ib2xdID0gbnVsbDtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBleGlzdGluZ1Rhc2suem9uZS5jYW5jZWxUYXNrKGV4aXN0aW5nVGFzayk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHJldHVyblRhcmdldCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gdGFyZ2V0O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgLy8gaXNzdWUgOTMwLCBkaWRuJ3QgZmluZCB0aGUgZXZlbnQgbmFtZSBvciBjYWxsYmFja1xuICAgICAgICAgICAgICAgIC8vIGZyb20gem9uZSBrZXB0IGV4aXN0aW5nVGFza3MsIHRoZSBjYWxsYmFjayBtYXliZVxuICAgICAgICAgICAgICAgIC8vIGFkZGVkIG91dHNpZGUgb2Ygem9uZSwgd2UgbmVlZCB0byBjYWxsIG5hdGl2ZSByZW1vdmVFdmVudExpc3RlbmVyXG4gICAgICAgICAgICAgICAgLy8gdG8gdHJ5IHRvIHJlbW92ZSBpdC5cbiAgICAgICAgICAgICAgICByZXR1cm4gbmF0aXZlUmVtb3ZlRXZlbnRMaXN0ZW5lci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHByb3RvW0xJU1RFTkVSU19FVkVOVF9MSVNURU5FUl0gPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgdmFyIHRhcmdldCA9IHRoaXMgfHwgX2dsb2JhbDtcbiAgICAgICAgICAgICAgICB2YXIgZXZlbnROYW1lID0gYXJndW1lbnRzWzBdO1xuICAgICAgICAgICAgICAgIGlmIChwYXRjaE9wdGlvbnMgJiYgcGF0Y2hPcHRpb25zLnRyYW5zZmVyRXZlbnROYW1lKSB7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50TmFtZSA9IHBhdGNoT3B0aW9ucy50cmFuc2ZlckV2ZW50TmFtZShldmVudE5hbWUpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB2YXIgbGlzdGVuZXJzID0gW107XG4gICAgICAgICAgICAgICAgdmFyIHRhc2tzID0gZmluZEV2ZW50VGFza3ModGFyZ2V0LCBldmVudE5hbWVUb1N0cmluZyA/IGV2ZW50TmFtZVRvU3RyaW5nKGV2ZW50TmFtZSkgOiBldmVudE5hbWUpO1xuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGFza3MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHRhc2sgPSB0YXNrc1tpXTtcbiAgICAgICAgICAgICAgICAgICAgdmFyIGRlbGVnYXRlID0gdGFzay5vcmlnaW5hbERlbGVnYXRlID8gdGFzay5vcmlnaW5hbERlbGVnYXRlIDogdGFzay5jYWxsYmFjaztcbiAgICAgICAgICAgICAgICAgICAgbGlzdGVuZXJzLnB1c2goZGVsZWdhdGUpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gbGlzdGVuZXJzO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHByb3RvW1JFTU9WRV9BTExfTElTVEVORVJTX0VWRU5UX0xJU1RFTkVSXSA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICB2YXIgdGFyZ2V0ID0gdGhpcyB8fCBfZ2xvYmFsO1xuICAgICAgICAgICAgICAgIHZhciBldmVudE5hbWUgPSBhcmd1bWVudHNbMF07XG4gICAgICAgICAgICAgICAgaWYgKCFldmVudE5hbWUpIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIGtleXMgPSBPYmplY3Qua2V5cyh0YXJnZXQpO1xuICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGtleXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBwcm9wID0ga2V5c1tpXTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBtYXRjaCA9IEVWRU5UX05BTUVfU1lNQk9MX1JFR1guZXhlYyhwcm9wKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBldnROYW1lID0gbWF0Y2ggJiYgbWF0Y2hbMV07XG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBpbiBub2RlanMgRXZlbnRFbWl0dGVyLCByZW1vdmVMaXN0ZW5lciBldmVudCBpc1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gdXNlZCBmb3IgbW9uaXRvcmluZyB0aGUgcmVtb3ZlTGlzdGVuZXIgY2FsbCxcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHNvIGp1c3Qga2VlcCByZW1vdmVMaXN0ZW5lciBldmVudExpc3RlbmVyIHVudGlsXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBhbGwgb3RoZXIgZXZlbnRMaXN0ZW5lcnMgYXJlIHJlbW92ZWRcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChldnROYW1lICYmIGV2dE5hbWUgIT09ICdyZW1vdmVMaXN0ZW5lcicpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzW1JFTU9WRV9BTExfTElTVEVORVJTX0VWRU5UX0xJU1RFTkVSXS5jYWxsKHRoaXMsIGV2dE5hbWUpO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIC8vIHJlbW92ZSByZW1vdmVMaXN0ZW5lciBsaXN0ZW5lciBmaW5hbGx5XG4gICAgICAgICAgICAgICAgICAgIHRoaXNbUkVNT1ZFX0FMTF9MSVNURU5FUlNfRVZFTlRfTElTVEVORVJdLmNhbGwodGhpcywgJ3JlbW92ZUxpc3RlbmVyJyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBpZiAocGF0Y2hPcHRpb25zICYmIHBhdGNoT3B0aW9ucy50cmFuc2ZlckV2ZW50TmFtZSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnROYW1lID0gcGF0Y2hPcHRpb25zLnRyYW5zZmVyRXZlbnROYW1lKGV2ZW50TmFtZSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgdmFyIHN5bWJvbEV2ZW50TmFtZXMgPSB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdO1xuICAgICAgICAgICAgICAgICAgICBpZiAoc3ltYm9sRXZlbnROYW1lcykge1xuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHN5bWJvbEV2ZW50TmFtZSA9IHN5bWJvbEV2ZW50TmFtZXNbRkFMU0VfU1RSXTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBzeW1ib2xDYXB0dXJlRXZlbnROYW1lID0gc3ltYm9sRXZlbnROYW1lc1tUUlVFX1NUUl07XG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgdGFza3MgPSB0YXJnZXRbc3ltYm9sRXZlbnROYW1lXTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBjYXB0dXJlVGFza3MgPSB0YXJnZXRbc3ltYm9sQ2FwdHVyZUV2ZW50TmFtZV07XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodGFza3MpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgcmVtb3ZlVGFza3MgPSB0YXNrcy5zbGljZSgpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcmVtb3ZlVGFza3MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHRhc2sgPSByZW1vdmVUYXNrc1tpXTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGRlbGVnYXRlID0gdGFzay5vcmlnaW5hbERlbGVnYXRlID8gdGFzay5vcmlnaW5hbERlbGVnYXRlIDogdGFzay5jYWxsYmFjaztcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpc1tSRU1PVkVfRVZFTlRfTElTVEVORVJdLmNhbGwodGhpcywgZXZlbnROYW1lLCBkZWxlZ2F0ZSwgdGFzay5vcHRpb25zKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoY2FwdHVyZVRhc2tzKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHJlbW92ZVRhc2tzID0gY2FwdHVyZVRhc2tzLnNsaWNlKCk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCByZW1vdmVUYXNrcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgdGFzayA9IHJlbW92ZVRhc2tzW2ldO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgZGVsZWdhdGUgPSB0YXNrLm9yaWdpbmFsRGVsZWdhdGUgPyB0YXNrLm9yaWdpbmFsRGVsZWdhdGUgOiB0YXNrLmNhbGxiYWNrO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzW1JFTU9WRV9FVkVOVF9MSVNURU5FUl0uY2FsbCh0aGlzLCBldmVudE5hbWUsIGRlbGVnYXRlLCB0YXNrLm9wdGlvbnMpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBpZiAocmV0dXJuVGFyZ2V0KSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICAvLyBmb3IgbmF0aXZlIHRvU3RyaW5nIHBhdGNoXG4gICAgICAgICAgICBhdHRhY2hPcmlnaW5Ub1BhdGNoZWQocHJvdG9bQUREX0VWRU5UX0xJU1RFTkVSXSwgbmF0aXZlQWRkRXZlbnRMaXN0ZW5lcik7XG4gICAgICAgICAgICBhdHRhY2hPcmlnaW5Ub1BhdGNoZWQocHJvdG9bUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXSwgbmF0aXZlUmVtb3ZlRXZlbnRMaXN0ZW5lcik7XG4gICAgICAgICAgICBpZiAobmF0aXZlUmVtb3ZlQWxsTGlzdGVuZXJzKSB7XG4gICAgICAgICAgICAgICAgYXR0YWNoT3JpZ2luVG9QYXRjaGVkKHByb3RvW1JFTU9WRV9BTExfTElTVEVORVJTX0VWRU5UX0xJU1RFTkVSXSwgbmF0aXZlUmVtb3ZlQWxsTGlzdGVuZXJzKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmIChuYXRpdmVMaXN0ZW5lcnMpIHtcbiAgICAgICAgICAgICAgICBhdHRhY2hPcmlnaW5Ub1BhdGNoZWQocHJvdG9bTElTVEVORVJTX0VWRU5UX0xJU1RFTkVSXSwgbmF0aXZlTGlzdGVuZXJzKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIHZhciByZXN1bHRzID0gW107XG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXBpcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgcmVzdWx0c1tpXSA9IHBhdGNoRXZlbnRUYXJnZXRNZXRob2RzKGFwaXNbaV0sIHBhdGNoT3B0aW9ucyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlc3VsdHM7XG4gICAgfVxuICAgIGZ1bmN0aW9uIGZpbmRFdmVudFRhc2tzKHRhcmdldCwgZXZlbnROYW1lKSB7XG4gICAgICAgIGlmICghZXZlbnROYW1lKSB7XG4gICAgICAgICAgICB2YXIgZm91bmRUYXNrcyA9IFtdO1xuICAgICAgICAgICAgZm9yICh2YXIgcHJvcCBpbiB0YXJnZXQpIHtcbiAgICAgICAgICAgICAgICB2YXIgbWF0Y2ggPSBFVkVOVF9OQU1FX1NZTUJPTF9SRUdYLmV4ZWMocHJvcCk7XG4gICAgICAgICAgICAgICAgdmFyIGV2dE5hbWUgPSBtYXRjaCAmJiBtYXRjaFsxXTtcbiAgICAgICAgICAgICAgICBpZiAoZXZ0TmFtZSAmJiAoIWV2ZW50TmFtZSB8fCBldnROYW1lID09PSBldmVudE5hbWUpKSB7XG4gICAgICAgICAgICAgICAgICAgIHZhciB0YXNrcyA9IHRhcmdldFtwcm9wXTtcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRhc2tzKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRhc2tzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZm91bmRUYXNrcy5wdXNoKHRhc2tzW2ldKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBmb3VuZFRhc2tzO1xuICAgICAgICB9XG4gICAgICAgIHZhciBzeW1ib2xFdmVudE5hbWUgPSB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdO1xuICAgICAgICBpZiAoIXN5bWJvbEV2ZW50TmFtZSkge1xuICAgICAgICAgICAgcHJlcGFyZUV2ZW50TmFtZXMoZXZlbnROYW1lKTtcbiAgICAgICAgICAgIHN5bWJvbEV2ZW50TmFtZSA9IHpvbmVTeW1ib2xFdmVudE5hbWVzW2V2ZW50TmFtZV07XG4gICAgICAgIH1cbiAgICAgICAgdmFyIGNhcHR1cmVGYWxzZVRhc2tzID0gdGFyZ2V0W3N5bWJvbEV2ZW50TmFtZVtGQUxTRV9TVFJdXTtcbiAgICAgICAgdmFyIGNhcHR1cmVUcnVlVGFza3MgPSB0YXJnZXRbc3ltYm9sRXZlbnROYW1lW1RSVUVfU1RSXV07XG4gICAgICAgIGlmICghY2FwdHVyZUZhbHNlVGFza3MpIHtcbiAgICAgICAgICAgIHJldHVybiBjYXB0dXJlVHJ1ZVRhc2tzID8gY2FwdHVyZVRydWVUYXNrcy5zbGljZSgpIDogW107XG4gICAgICAgIH1cbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gY2FwdHVyZVRydWVUYXNrcyA/IGNhcHR1cmVGYWxzZVRhc2tzLmNvbmNhdChjYXB0dXJlVHJ1ZVRhc2tzKSA6XG4gICAgICAgICAgICAgICAgY2FwdHVyZUZhbHNlVGFza3Muc2xpY2UoKTtcbiAgICAgICAgfVxuICAgIH1cbiAgICBmdW5jdGlvbiBwYXRjaEV2ZW50UHJvdG90eXBlKGdsb2JhbCwgYXBpKSB7XG4gICAgICAgIHZhciBFdmVudCA9IGdsb2JhbFsnRXZlbnQnXTtcbiAgICAgICAgaWYgKEV2ZW50ICYmIEV2ZW50LnByb3RvdHlwZSkge1xuICAgICAgICAgICAgYXBpLnBhdGNoTWV0aG9kKEV2ZW50LnByb3RvdHlwZSwgJ3N0b3BJbW1lZGlhdGVQcm9wYWdhdGlvbicsIGZ1bmN0aW9uIChkZWxlZ2F0ZSkgeyByZXR1cm4gZnVuY3Rpb24gKHNlbGYsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICBzZWxmW0lNTUVESUFURV9QUk9QQUdBVElPTl9TWU1CT0xdID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAvLyB3ZSBuZWVkIHRvIGNhbGwgdGhlIG5hdGl2ZSBzdG9wSW1tZWRpYXRlUHJvcGFnYXRpb25cbiAgICAgICAgICAgICAgICAvLyBpbiBjYXNlIGluIHNvbWUgaHlicmlkIGFwcGxpY2F0aW9uLCBzb21lIHBhcnQgb2ZcbiAgICAgICAgICAgICAgICAvLyBhcHBsaWNhdGlvbiB3aWxsIGJlIGNvbnRyb2xsZWQgYnkgem9uZSwgc29tZSBhcmUgbm90XG4gICAgICAgICAgICAgICAgZGVsZWdhdGUgJiYgZGVsZWdhdGUuYXBwbHkoc2VsZiwgYXJncyk7XG4gICAgICAgICAgICB9OyB9KTtcbiAgICAgICAgfVxuICAgIH1cbiAgICAvKipcbiAgICAgKiBAbGljZW5zZVxuICAgICAqIENvcHlyaWdodCBHb29nbGUgTExDIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gICAgICpcbiAgICAgKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICAgICAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBwYXRjaENhbGxiYWNrcyhhcGksIHRhcmdldCwgdGFyZ2V0TmFtZSwgbWV0aG9kLCBjYWxsYmFja3MpIHtcbiAgICAgICAgdmFyIHN5bWJvbCA9IFpvbmUuX19zeW1ib2xfXyhtZXRob2QpO1xuICAgICAgICBpZiAodGFyZ2V0W3N5bWJvbF0pIHtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICB2YXIgbmF0aXZlRGVsZWdhdGUgPSB0YXJnZXRbc3ltYm9sXSA9IHRhcmdldFttZXRob2RdO1xuICAgICAgICB0YXJnZXRbbWV0aG9kXSA9IGZ1bmN0aW9uIChuYW1lLCBvcHRzLCBvcHRpb25zKSB7XG4gICAgICAgICAgICBpZiAob3B0cyAmJiBvcHRzLnByb3RvdHlwZSkge1xuICAgICAgICAgICAgICAgIGNhbGxiYWNrcy5mb3JFYWNoKGZ1bmN0aW9uIChjYWxsYmFjaykge1xuICAgICAgICAgICAgICAgICAgICB2YXIgc291cmNlID0gXCJcIi5jb25jYXQodGFyZ2V0TmFtZSwgXCIuXCIpLmNvbmNhdChtZXRob2QsIFwiOjpcIikgKyBjYWxsYmFjaztcbiAgICAgICAgICAgICAgICAgICAgdmFyIHByb3RvdHlwZSA9IG9wdHMucHJvdG90eXBlO1xuICAgICAgICAgICAgICAgICAgICAvLyBOb3RlOiB0aGUgYHBhdGNoQ2FsbGJhY2tzYCBpcyB1c2VkIGZvciBwYXRjaGluZyB0aGUgYGRvY3VtZW50LnJlZ2lzdGVyRWxlbWVudGAgYW5kXG4gICAgICAgICAgICAgICAgICAgIC8vIGBjdXN0b21FbGVtZW50cy5kZWZpbmVgLiBXZSBleHBsaWNpdGx5IHdyYXAgdGhlIHBhdGNoaW5nIGNvZGUgaW50byB0cnktY2F0Y2ggc2luY2VcbiAgICAgICAgICAgICAgICAgICAgLy8gY2FsbGJhY2tzIG1heSBiZSBhbHJlYWR5IHBhdGNoZWQgYnkgb3RoZXIgd2ViIGNvbXBvbmVudHMgZnJhbWV3b3JrcyAoZS5nLiBMV0MpLCBhbmQgdGhleVxuICAgICAgICAgICAgICAgICAgICAvLyBtYWtlIHRob3NlIHByb3BlcnRpZXMgbm9uLXdyaXRhYmxlLiBUaGlzIG1lYW5zIHRoYXQgcGF0Y2hpbmcgY2FsbGJhY2sgd2lsbCB0aHJvdyBhbiBlcnJvclxuICAgICAgICAgICAgICAgICAgICAvLyBgY2Fubm90IGFzc2lnbiB0byByZWFkLW9ubHkgcHJvcGVydHlgLiBTZWUgdGhpcyBjb2RlIGFzIGFuIGV4YW1wbGU6XG4gICAgICAgICAgICAgICAgICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9zYWxlc2ZvcmNlL2x3Yy9ibG9iL21hc3Rlci9wYWNrYWdlcy9AbHdjL2VuZ2luZS1jb3JlL3NyYy9mcmFtZXdvcmsvYmFzZS1icmlkZ2UtZWxlbWVudC50cyNMMTgwLUwxODZcbiAgICAgICAgICAgICAgICAgICAgLy8gV2UgZG9uJ3Qgd2FudCB0byBzdG9wIHRoZSBhcHBsaWNhdGlvbiByZW5kZXJpbmcgaWYgd2UgY291bGRuJ3QgcGF0Y2ggc29tZVxuICAgICAgICAgICAgICAgICAgICAvLyBjYWxsYmFjaywgZS5nLiBgYXR0cmlidXRlQ2hhbmdlZENhbGxiYWNrYC5cbiAgICAgICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm90b3R5cGUuaGFzT3duUHJvcGVydHkoY2FsbGJhY2spKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGRlc2NyaXB0b3IgPSBhcGkuT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHByb3RvdHlwZSwgY2FsbGJhY2spO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChkZXNjcmlwdG9yICYmIGRlc2NyaXB0b3IudmFsdWUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZGVzY3JpcHRvci52YWx1ZSA9IGFwaS53cmFwV2l0aEN1cnJlbnRab25lKGRlc2NyaXB0b3IudmFsdWUsIHNvdXJjZSk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFwaS5fcmVkZWZpbmVQcm9wZXJ0eShvcHRzLnByb3RvdHlwZSwgY2FsbGJhY2ssIGRlc2NyaXB0b3IpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChwcm90b3R5cGVbY2FsbGJhY2tdKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb3RvdHlwZVtjYWxsYmFja10gPSBhcGkud3JhcFdpdGhDdXJyZW50Wm9uZShwcm90b3R5cGVbY2FsbGJhY2tdLCBzb3VyY2UpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKHByb3RvdHlwZVtjYWxsYmFja10pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm90b3R5cGVbY2FsbGJhY2tdID0gYXBpLndyYXBXaXRoQ3VycmVudFpvbmUocHJvdG90eXBlW2NhbGxiYWNrXSwgc291cmNlKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBjYXRjaCAoX2EpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIE5vdGU6IHdlIGxlYXZlIHRoZSBjYXRjaCBibG9jayBlbXB0eSBzaW5jZSB0aGVyZSdzIG5vIHdheSB0byBoYW5kbGUgdGhlIGVycm9yIHJlbGF0ZWRcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHRvIG5vbi13cml0YWJsZSBwcm9wZXJ0eS5cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIG5hdGl2ZURlbGVnYXRlLmNhbGwodGFyZ2V0LCBuYW1lLCBvcHRzLCBvcHRpb25zKTtcbiAgICAgICAgfTtcbiAgICAgICAgYXBpLmF0dGFjaE9yaWdpblRvUGF0Y2hlZCh0YXJnZXRbbWV0aG9kXSwgbmF0aXZlRGVsZWdhdGUpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBAbGljZW5zZVxuICAgICAqIENvcHlyaWdodCBHb29nbGUgTExDIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gICAgICpcbiAgICAgKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICAgICAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBmaWx0ZXJQcm9wZXJ0aWVzKHRhcmdldCwgb25Qcm9wZXJ0aWVzLCBpZ25vcmVQcm9wZXJ0aWVzKSB7XG4gICAgICAgIGlmICghaWdub3JlUHJvcGVydGllcyB8fCBpZ25vcmVQcm9wZXJ0aWVzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAgICAgcmV0dXJuIG9uUHJvcGVydGllcztcbiAgICAgICAgfVxuICAgICAgICB2YXIgdGlwID0gaWdub3JlUHJvcGVydGllcy5maWx0ZXIoZnVuY3Rpb24gKGlwKSB7IHJldHVybiBpcC50YXJnZXQgPT09IHRhcmdldDsgfSk7XG4gICAgICAgIGlmICghdGlwIHx8IHRpcC5sZW5ndGggPT09IDApIHtcbiAgICAgICAgICAgIHJldHVybiBvblByb3BlcnRpZXM7XG4gICAgICAgIH1cbiAgICAgICAgdmFyIHRhcmdldElnbm9yZVByb3BlcnRpZXMgPSB0aXBbMF0uaWdub3JlUHJvcGVydGllcztcbiAgICAgICAgcmV0dXJuIG9uUHJvcGVydGllcy5maWx0ZXIoZnVuY3Rpb24gKG9wKSB7IHJldHVybiB0YXJnZXRJZ25vcmVQcm9wZXJ0aWVzLmluZGV4T2Yob3ApID09PSAtMTsgfSk7XG4gICAgfVxuICAgIGZ1bmN0aW9uIHBhdGNoRmlsdGVyZWRQcm9wZXJ0aWVzKHRhcmdldCwgb25Qcm9wZXJ0aWVzLCBpZ25vcmVQcm9wZXJ0aWVzLCBwcm90b3R5cGUpIHtcbiAgICAgICAgLy8gY2hlY2sgd2hldGhlciB0YXJnZXQgaXMgYXZhaWxhYmxlLCBzb21ldGltZXMgdGFyZ2V0IHdpbGwgYmUgdW5kZWZpbmVkXG4gICAgICAgIC8vIGJlY2F1c2UgZGlmZmVyZW50IGJyb3dzZXIgb3Igc29tZSAzcmQgcGFydHkgcGx1Z2luLlxuICAgICAgICBpZiAoIXRhcmdldCkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHZhciBmaWx0ZXJlZFByb3BlcnRpZXMgPSBmaWx0ZXJQcm9wZXJ0aWVzKHRhcmdldCwgb25Qcm9wZXJ0aWVzLCBpZ25vcmVQcm9wZXJ0aWVzKTtcbiAgICAgICAgcGF0Y2hPblByb3BlcnRpZXModGFyZ2V0LCBmaWx0ZXJlZFByb3BlcnRpZXMsIHByb3RvdHlwZSk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEdldCBhbGwgZXZlbnQgbmFtZSBwcm9wZXJ0aWVzIHdoaWNoIHRoZSBldmVudCBuYW1lIHN0YXJ0c1dpdGggYG9uYFxuICAgICAqIGZyb20gdGhlIHRhcmdldCBvYmplY3QgaXRzZWxmLCBpbmhlcml0ZWQgcHJvcGVydGllcyBhcmUgbm90IGNvbnNpZGVyZWQuXG4gICAgICovXG4gICAgZnVuY3Rpb24gZ2V0T25FdmVudE5hbWVzKHRhcmdldCkge1xuICAgICAgICByZXR1cm4gT2JqZWN0LmdldE93blByb3BlcnR5TmFtZXModGFyZ2V0KVxuICAgICAgICAgICAgLmZpbHRlcihmdW5jdGlvbiAobmFtZSkgeyByZXR1cm4gbmFtZS5zdGFydHNXaXRoKCdvbicpICYmIG5hbWUubGVuZ3RoID4gMjsgfSlcbiAgICAgICAgICAgIC5tYXAoZnVuY3Rpb24gKG5hbWUpIHsgcmV0dXJuIG5hbWUuc3Vic3RyaW5nKDIpOyB9KTtcbiAgICB9XG4gICAgZnVuY3Rpb24gcHJvcGVydHlEZXNjcmlwdG9yUGF0Y2goYXBpLCBfZ2xvYmFsKSB7XG4gICAgICAgIGlmIChpc05vZGUgJiYgIWlzTWl4KSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgaWYgKFpvbmVbYXBpLnN5bWJvbCgncGF0Y2hFdmVudHMnKV0pIHtcbiAgICAgICAgICAgIC8vIGV2ZW50cyBhcmUgYWxyZWFkeSBiZWVuIHBhdGNoZWQgYnkgbGVnYWN5IHBhdGNoLlxuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHZhciBpZ25vcmVQcm9wZXJ0aWVzID0gX2dsb2JhbFsnX19ab25lX2lnbm9yZV9vbl9wcm9wZXJ0aWVzJ107XG4gICAgICAgIC8vIGZvciBicm93c2VycyB0aGF0IHdlIGNhbiBwYXRjaCB0aGUgZGVzY3JpcHRvcjogIENocm9tZSAmIEZpcmVmb3hcbiAgICAgICAgdmFyIHBhdGNoVGFyZ2V0cyA9IFtdO1xuICAgICAgICBpZiAoaXNCcm93c2VyKSB7XG4gICAgICAgICAgICB2YXIgaW50ZXJuYWxXaW5kb3dfMSA9IHdpbmRvdztcbiAgICAgICAgICAgIHBhdGNoVGFyZ2V0cyA9IHBhdGNoVGFyZ2V0cy5jb25jYXQoW1xuICAgICAgICAgICAgICAgICdEb2N1bWVudCcsICdTVkdFbGVtZW50JywgJ0VsZW1lbnQnLCAnSFRNTEVsZW1lbnQnLCAnSFRNTEJvZHlFbGVtZW50JywgJ0hUTUxNZWRpYUVsZW1lbnQnLFxuICAgICAgICAgICAgICAgICdIVE1MRnJhbWVTZXRFbGVtZW50JywgJ0hUTUxGcmFtZUVsZW1lbnQnLCAnSFRNTElGcmFtZUVsZW1lbnQnLCAnSFRNTE1hcnF1ZWVFbGVtZW50JywgJ1dvcmtlcidcbiAgICAgICAgICAgIF0pO1xuICAgICAgICAgICAgdmFyIGlnbm9yZUVycm9yUHJvcGVydGllcyA9IGlzSUUoKSA/IFt7IHRhcmdldDogaW50ZXJuYWxXaW5kb3dfMSwgaWdub3JlUHJvcGVydGllczogWydlcnJvciddIH1dIDogW107XG4gICAgICAgICAgICAvLyBpbiBJRS9FZGdlLCBvblByb3Agbm90IGV4aXN0IGluIHdpbmRvdyBvYmplY3QsIGJ1dCBpbiBXaW5kb3dQcm90b3R5cGVcbiAgICAgICAgICAgIC8vIHNvIHdlIG5lZWQgdG8gcGFzcyBXaW5kb3dQcm90b3R5cGUgdG8gY2hlY2sgb25Qcm9wIGV4aXN0IG9yIG5vdFxuICAgICAgICAgICAgcGF0Y2hGaWx0ZXJlZFByb3BlcnRpZXMoaW50ZXJuYWxXaW5kb3dfMSwgZ2V0T25FdmVudE5hbWVzKGludGVybmFsV2luZG93XzEpLCBpZ25vcmVQcm9wZXJ0aWVzID8gaWdub3JlUHJvcGVydGllcy5jb25jYXQoaWdub3JlRXJyb3JQcm9wZXJ0aWVzKSA6IGlnbm9yZVByb3BlcnRpZXMsIE9iamVjdEdldFByb3RvdHlwZU9mKGludGVybmFsV2luZG93XzEpKTtcbiAgICAgICAgfVxuICAgICAgICBwYXRjaFRhcmdldHMgPSBwYXRjaFRhcmdldHMuY29uY2F0KFtcbiAgICAgICAgICAgICdYTUxIdHRwUmVxdWVzdCcsICdYTUxIdHRwUmVxdWVzdEV2ZW50VGFyZ2V0JywgJ0lEQkluZGV4JywgJ0lEQlJlcXVlc3QnLCAnSURCT3BlbkRCUmVxdWVzdCcsXG4gICAgICAgICAgICAnSURCRGF0YWJhc2UnLCAnSURCVHJhbnNhY3Rpb24nLCAnSURCQ3Vyc29yJywgJ1dlYlNvY2tldCdcbiAgICAgICAgXSk7XG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgcGF0Y2hUYXJnZXRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICB2YXIgdGFyZ2V0ID0gX2dsb2JhbFtwYXRjaFRhcmdldHNbaV1dO1xuICAgICAgICAgICAgdGFyZ2V0ICYmIHRhcmdldC5wcm90b3R5cGUgJiZcbiAgICAgICAgICAgICAgICBwYXRjaEZpbHRlcmVkUHJvcGVydGllcyh0YXJnZXQucHJvdG90eXBlLCBnZXRPbkV2ZW50TmFtZXModGFyZ2V0LnByb3RvdHlwZSksIGlnbm9yZVByb3BlcnRpZXMpO1xuICAgICAgICB9XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCd1dGlsJywgZnVuY3Rpb24gKGdsb2JhbCwgWm9uZSwgYXBpKSB7XG4gICAgICAgIC8vIENvbGxlY3QgbmF0aXZlIGV2ZW50IG5hbWVzIGJ5IGxvb2tpbmcgYXQgcHJvcGVydGllc1xuICAgICAgICAvLyBvbiB0aGUgZ2xvYmFsIG5hbWVzcGFjZSwgZS5nLiAnb25jbGljaycuXG4gICAgICAgIHZhciBldmVudE5hbWVzID0gZ2V0T25FdmVudE5hbWVzKGdsb2JhbCk7XG4gICAgICAgIGFwaS5wYXRjaE9uUHJvcGVydGllcyA9IHBhdGNoT25Qcm9wZXJ0aWVzO1xuICAgICAgICBhcGkucGF0Y2hNZXRob2QgPSBwYXRjaE1ldGhvZDtcbiAgICAgICAgYXBpLmJpbmRBcmd1bWVudHMgPSBiaW5kQXJndW1lbnRzO1xuICAgICAgICBhcGkucGF0Y2hNYWNyb1Rhc2sgPSBwYXRjaE1hY3JvVGFzaztcbiAgICAgICAgLy8gSW4gZWFybGllciB2ZXJzaW9uIG9mIHpvbmUuanMgKDwwLjkuMCksIHdlIHVzZSBlbnYgbmFtZSBgX196b25lX3N5bWJvbF9fQkxBQ0tfTElTVEVEX0VWRU5UU2AgdG9cbiAgICAgICAgLy8gZGVmaW5lIHdoaWNoIGV2ZW50cyB3aWxsIG5vdCBiZSBwYXRjaGVkIGJ5IGBab25lLmpzYC5cbiAgICAgICAgLy8gSW4gbmV3ZXIgdmVyc2lvbiAoPj0wLjkuMCksIHdlIGNoYW5nZSB0aGUgZW52IG5hbWUgdG8gYF9fem9uZV9zeW1ib2xfX1VOUEFUQ0hFRF9FVkVOVFNgIHRvIGtlZXBcbiAgICAgICAgLy8gdGhlIG5hbWUgY29uc2lzdGVudCB3aXRoIGFuZ3VsYXIgcmVwby5cbiAgICAgICAgLy8gVGhlICBgX196b25lX3N5bWJvbF9fQkxBQ0tfTElTVEVEX0VWRU5UU2AgaXMgZGVwcmVjYXRlZCwgYnV0IGl0IGlzIHN0aWxsIGJlIHN1cHBvcnRlZCBmb3JcbiAgICAgICAgLy8gYmFja3dhcmRzIGNvbXBhdGliaWxpdHkuXG4gICAgICAgIHZhciBTWU1CT0xfQkxBQ0tfTElTVEVEX0VWRU5UUyA9IFpvbmUuX19zeW1ib2xfXygnQkxBQ0tfTElTVEVEX0VWRU5UUycpO1xuICAgICAgICB2YXIgU1lNQk9MX1VOUEFUQ0hFRF9FVkVOVFMgPSBab25lLl9fc3ltYm9sX18oJ1VOUEFUQ0hFRF9FVkVOVFMnKTtcbiAgICAgICAgaWYgKGdsb2JhbFtTWU1CT0xfVU5QQVRDSEVEX0VWRU5UU10pIHtcbiAgICAgICAgICAgIGdsb2JhbFtTWU1CT0xfQkxBQ0tfTElTVEVEX0VWRU5UU10gPSBnbG9iYWxbU1lNQk9MX1VOUEFUQ0hFRF9FVkVOVFNdO1xuICAgICAgICB9XG4gICAgICAgIGlmIChnbG9iYWxbU1lNQk9MX0JMQUNLX0xJU1RFRF9FVkVOVFNdKSB7XG4gICAgICAgICAgICBab25lW1NZTUJPTF9CTEFDS19MSVNURURfRVZFTlRTXSA9IFpvbmVbU1lNQk9MX1VOUEFUQ0hFRF9FVkVOVFNdID1cbiAgICAgICAgICAgICAgICBnbG9iYWxbU1lNQk9MX0JMQUNLX0xJU1RFRF9FVkVOVFNdO1xuICAgICAgICB9XG4gICAgICAgIGFwaS5wYXRjaEV2ZW50UHJvdG90eXBlID0gcGF0Y2hFdmVudFByb3RvdHlwZTtcbiAgICAgICAgYXBpLnBhdGNoRXZlbnRUYXJnZXQgPSBwYXRjaEV2ZW50VGFyZ2V0O1xuICAgICAgICBhcGkuaXNJRU9yRWRnZSA9IGlzSUVPckVkZ2U7XG4gICAgICAgIGFwaS5PYmplY3REZWZpbmVQcm9wZXJ0eSA9IE9iamVjdERlZmluZVByb3BlcnR5O1xuICAgICAgICBhcGkuT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yID0gT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yO1xuICAgICAgICBhcGkuT2JqZWN0Q3JlYXRlID0gT2JqZWN0Q3JlYXRlO1xuICAgICAgICBhcGkuQXJyYXlTbGljZSA9IEFycmF5U2xpY2U7XG4gICAgICAgIGFwaS5wYXRjaENsYXNzID0gcGF0Y2hDbGFzcztcbiAgICAgICAgYXBpLndyYXBXaXRoQ3VycmVudFpvbmUgPSB3cmFwV2l0aEN1cnJlbnRab25lO1xuICAgICAgICBhcGkuZmlsdGVyUHJvcGVydGllcyA9IGZpbHRlclByb3BlcnRpZXM7XG4gICAgICAgIGFwaS5hdHRhY2hPcmlnaW5Ub1BhdGNoZWQgPSBhdHRhY2hPcmlnaW5Ub1BhdGNoZWQ7XG4gICAgICAgIGFwaS5fcmVkZWZpbmVQcm9wZXJ0eSA9IE9iamVjdC5kZWZpbmVQcm9wZXJ0eTtcbiAgICAgICAgYXBpLnBhdGNoQ2FsbGJhY2tzID0gcGF0Y2hDYWxsYmFja3M7XG4gICAgICAgIGFwaS5nZXRHbG9iYWxPYmplY3RzID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gKHtcbiAgICAgICAgICAgIGdsb2JhbFNvdXJjZXM6IGdsb2JhbFNvdXJjZXMsXG4gICAgICAgICAgICB6b25lU3ltYm9sRXZlbnROYW1lczogem9uZVN5bWJvbEV2ZW50TmFtZXMsXG4gICAgICAgICAgICBldmVudE5hbWVzOiBldmVudE5hbWVzLFxuICAgICAgICAgICAgaXNCcm93c2VyOiBpc0Jyb3dzZXIsXG4gICAgICAgICAgICBpc01peDogaXNNaXgsXG4gICAgICAgICAgICBpc05vZGU6IGlzTm9kZSxcbiAgICAgICAgICAgIFRSVUVfU1RSOiBUUlVFX1NUUixcbiAgICAgICAgICAgIEZBTFNFX1NUUjogRkFMU0VfU1RSLFxuICAgICAgICAgICAgWk9ORV9TWU1CT0xfUFJFRklYOiBaT05FX1NZTUJPTF9QUkVGSVgsXG4gICAgICAgICAgICBBRERfRVZFTlRfTElTVEVORVJfU1RSOiBBRERfRVZFTlRfTElTVEVORVJfU1RSLFxuICAgICAgICAgICAgUkVNT1ZFX0VWRU5UX0xJU1RFTkVSX1NUUjogUkVNT1ZFX0VWRU5UX0xJU1RFTkVSX1NUUlxuICAgICAgICB9KTsgfTtcbiAgICB9KTtcbiAgICAvKipcbiAgICAgKiBAbGljZW5zZVxuICAgICAqIENvcHlyaWdodCBHb29nbGUgTExDIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gICAgICpcbiAgICAgKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICAgICAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAgICAgKi9cbiAgICAvKlxuICAgICAqIFRoaXMgaXMgbmVjZXNzYXJ5IGZvciBDaHJvbWUgYW5kIENocm9tZSBtb2JpbGUsIHRvIGVuYWJsZVxuICAgICAqIHRoaW5ncyBsaWtlIHJlZGVmaW5pbmcgYGNyZWF0ZWRDYWxsYmFja2Agb24gYW4gZWxlbWVudC5cbiAgICAgKi9cbiAgICB2YXIgem9uZVN5bWJvbDtcbiAgICB2YXIgX2RlZmluZVByb3BlcnR5O1xuICAgIHZhciBfZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yO1xuICAgIHZhciBfY3JlYXRlO1xuICAgIHZhciB1bmNvbmZpZ3VyYWJsZXNLZXk7XG4gICAgZnVuY3Rpb24gcHJvcGVydHlQYXRjaCgpIHtcbiAgICAgICAgem9uZVN5bWJvbCA9IFpvbmUuX19zeW1ib2xfXztcbiAgICAgICAgX2RlZmluZVByb3BlcnR5ID0gT2JqZWN0W3pvbmVTeW1ib2woJ2RlZmluZVByb3BlcnR5JyldID0gT2JqZWN0LmRlZmluZVByb3BlcnR5O1xuICAgICAgICBfZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yID0gT2JqZWN0W3pvbmVTeW1ib2woJ2dldE93blByb3BlcnR5RGVzY3JpcHRvcicpXSA9XG4gICAgICAgICAgICBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yO1xuICAgICAgICBfY3JlYXRlID0gT2JqZWN0LmNyZWF0ZTtcbiAgICAgICAgdW5jb25maWd1cmFibGVzS2V5ID0gem9uZVN5bWJvbCgndW5jb25maWd1cmFibGVzJyk7XG4gICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eSA9IGZ1bmN0aW9uIChvYmosIHByb3AsIGRlc2MpIHtcbiAgICAgICAgICAgIGlmIChpc1VuY29uZmlndXJhYmxlKG9iaiwgcHJvcCkpIHtcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdDYW5ub3QgYXNzaWduIHRvIHJlYWQgb25seSBwcm9wZXJ0eSBcXCcnICsgcHJvcCArICdcXCcgb2YgJyArIG9iaik7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgb3JpZ2luYWxDb25maWd1cmFibGVGbGFnID0gZGVzYy5jb25maWd1cmFibGU7XG4gICAgICAgICAgICBpZiAocHJvcCAhPT0gJ3Byb3RvdHlwZScpIHtcbiAgICAgICAgICAgICAgICBkZXNjID0gcmV3cml0ZURlc2NyaXB0b3Iob2JqLCBwcm9wLCBkZXNjKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBfdHJ5RGVmaW5lUHJvcGVydHkob2JqLCBwcm9wLCBkZXNjLCBvcmlnaW5hbENvbmZpZ3VyYWJsZUZsYWcpO1xuICAgICAgICB9O1xuICAgICAgICBPYmplY3QuZGVmaW5lUHJvcGVydGllcyA9IGZ1bmN0aW9uIChvYmosIHByb3BzKSB7XG4gICAgICAgICAgICBPYmplY3Qua2V5cyhwcm9wcykuZm9yRWFjaChmdW5jdGlvbiAocHJvcCkge1xuICAgICAgICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShvYmosIHByb3AsIHByb3BzW3Byb3BdKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZm9yICh2YXIgX2kgPSAwLCBfYiA9IE9iamVjdC5nZXRPd25Qcm9wZXJ0eVN5bWJvbHMocHJvcHMpOyBfaSA8IF9iLmxlbmd0aDsgX2krKykge1xuICAgICAgICAgICAgICAgIHZhciBzeW0gPSBfYltfaV07XG4gICAgICAgICAgICAgICAgdmFyIGRlc2MgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHByb3BzLCBzeW0pO1xuICAgICAgICAgICAgICAgIC8vIFNpbmNlIGBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzYCByZXR1cm5zICphbGwqIHN5bWJvbHMsXG4gICAgICAgICAgICAgICAgLy8gaW5jbHVkaW5nIG5vbi1lbnVtZXJhYmxlIG9uZXMsIHJldHJpZXZlIHByb3BlcnR5IGRlc2NyaXB0b3IgYW5kIGNoZWNrXG4gICAgICAgICAgICAgICAgLy8gZW51bWVyYWJpbGl0eSB0aGVyZS4gUHJvY2VlZCB3aXRoIHRoZSByZXdyaXRlIG9ubHkgd2hlbiBhIHByb3BlcnR5IGlzXG4gICAgICAgICAgICAgICAgLy8gZW51bWVyYWJsZSB0byBtYWtlIHRoZSBsb2dpYyBjb25zaXN0ZW50IHdpdGggdGhlIHdheSByZWd1bGFyXG4gICAgICAgICAgICAgICAgLy8gcHJvcGVydGllcyBhcmUgcmV0cmlldmVkICh2aWEgYE9iamVjdC5rZXlzYCwgd2hpY2ggcmVzcGVjdHNcbiAgICAgICAgICAgICAgICAvLyBgZW51bWVyYWJsZTogZmFsc2VgIGZsYWcpLiBNb3JlIGluZm9ybWF0aW9uOlxuICAgICAgICAgICAgICAgIC8vIGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0phdmFTY3JpcHQvRW51bWVyYWJpbGl0eV9hbmRfb3duZXJzaGlwX29mX3Byb3BlcnRpZXMjcmV0cmlldmFsXG4gICAgICAgICAgICAgICAgaWYgKGRlc2MgPT09IG51bGwgfHwgZGVzYyA9PT0gdm9pZCAwID8gdm9pZCAwIDogZGVzYy5lbnVtZXJhYmxlKSB7XG4gICAgICAgICAgICAgICAgICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShvYmosIHN5bSwgcHJvcHNbc3ltXSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIG9iajtcbiAgICAgICAgfTtcbiAgICAgICAgT2JqZWN0LmNyZWF0ZSA9IGZ1bmN0aW9uIChwcm90bywgcHJvcGVydGllc09iamVjdCkge1xuICAgICAgICAgICAgaWYgKHR5cGVvZiBwcm9wZXJ0aWVzT2JqZWN0ID09PSAnb2JqZWN0JyAmJiAhT2JqZWN0LmlzRnJvemVuKHByb3BlcnRpZXNPYmplY3QpKSB7XG4gICAgICAgICAgICAgICAgT2JqZWN0LmtleXMocHJvcGVydGllc09iamVjdCkuZm9yRWFjaChmdW5jdGlvbiAocHJvcCkge1xuICAgICAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzT2JqZWN0W3Byb3BdID0gcmV3cml0ZURlc2NyaXB0b3IocHJvdG8sIHByb3AsIHByb3BlcnRpZXNPYmplY3RbcHJvcF0pO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIF9jcmVhdGUocHJvdG8sIHByb3BlcnRpZXNPYmplY3QpO1xuICAgICAgICB9O1xuICAgICAgICBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yID0gZnVuY3Rpb24gKG9iaiwgcHJvcCkge1xuICAgICAgICAgICAgdmFyIGRlc2MgPSBfZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKG9iaiwgcHJvcCk7XG4gICAgICAgICAgICBpZiAoZGVzYyAmJiBpc1VuY29uZmlndXJhYmxlKG9iaiwgcHJvcCkpIHtcbiAgICAgICAgICAgICAgICBkZXNjLmNvbmZpZ3VyYWJsZSA9IGZhbHNlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIGRlc2M7XG4gICAgICAgIH07XG4gICAgfVxuICAgIGZ1bmN0aW9uIF9yZWRlZmluZVByb3BlcnR5KG9iaiwgcHJvcCwgZGVzYykge1xuICAgICAgICB2YXIgb3JpZ2luYWxDb25maWd1cmFibGVGbGFnID0gZGVzYy5jb25maWd1cmFibGU7XG4gICAgICAgIGRlc2MgPSByZXdyaXRlRGVzY3JpcHRvcihvYmosIHByb3AsIGRlc2MpO1xuICAgICAgICByZXR1cm4gX3RyeURlZmluZVByb3BlcnR5KG9iaiwgcHJvcCwgZGVzYywgb3JpZ2luYWxDb25maWd1cmFibGVGbGFnKTtcbiAgICB9XG4gICAgZnVuY3Rpb24gaXNVbmNvbmZpZ3VyYWJsZShvYmosIHByb3ApIHtcbiAgICAgICAgcmV0dXJuIG9iaiAmJiBvYmpbdW5jb25maWd1cmFibGVzS2V5XSAmJiBvYmpbdW5jb25maWd1cmFibGVzS2V5XVtwcm9wXTtcbiAgICB9XG4gICAgZnVuY3Rpb24gcmV3cml0ZURlc2NyaXB0b3Iob2JqLCBwcm9wLCBkZXNjKSB7XG4gICAgICAgIC8vIGlzc3VlLTkyNywgaWYgdGhlIGRlc2MgaXMgZnJvemVuLCBkb24ndCB0cnkgdG8gY2hhbmdlIHRoZSBkZXNjXG4gICAgICAgIGlmICghT2JqZWN0LmlzRnJvemVuKGRlc2MpKSB7XG4gICAgICAgICAgICBkZXNjLmNvbmZpZ3VyYWJsZSA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKCFkZXNjLmNvbmZpZ3VyYWJsZSkge1xuICAgICAgICAgICAgLy8gaXNzdWUtOTI3LCBpZiB0aGUgb2JqIGlzIGZyb3plbiwgZG9uJ3QgdHJ5IHRvIHNldCB0aGUgZGVzYyB0byBvYmpcbiAgICAgICAgICAgIGlmICghb2JqW3VuY29uZmlndXJhYmxlc0tleV0gJiYgIU9iamVjdC5pc0Zyb3plbihvYmopKSB7XG4gICAgICAgICAgICAgICAgX2RlZmluZVByb3BlcnR5KG9iaiwgdW5jb25maWd1cmFibGVzS2V5LCB7IHdyaXRhYmxlOiB0cnVlLCB2YWx1ZToge30gfSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAob2JqW3VuY29uZmlndXJhYmxlc0tleV0pIHtcbiAgICAgICAgICAgICAgICBvYmpbdW5jb25maWd1cmFibGVzS2V5XVtwcm9wXSA9IHRydWU7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGRlc2M7XG4gICAgfVxuICAgIGZ1bmN0aW9uIF90cnlEZWZpbmVQcm9wZXJ0eShvYmosIHByb3AsIGRlc2MsIG9yaWdpbmFsQ29uZmlndXJhYmxlRmxhZykge1xuICAgICAgICB0cnkge1xuICAgICAgICAgICAgcmV0dXJuIF9kZWZpbmVQcm9wZXJ0eShvYmosIHByb3AsIGRlc2MpO1xuICAgICAgICB9XG4gICAgICAgIGNhdGNoIChlcnJvcikge1xuICAgICAgICAgICAgaWYgKGRlc2MuY29uZmlndXJhYmxlKSB7XG4gICAgICAgICAgICAgICAgLy8gSW4gY2FzZSBvZiBlcnJvcnMsIHdoZW4gdGhlIGNvbmZpZ3VyYWJsZSBmbGFnIHdhcyBsaWtlbHkgc2V0IGJ5IHJld3JpdGVEZXNjcmlwdG9yKCksXG4gICAgICAgICAgICAgICAgLy8gbGV0J3MgcmV0cnkgd2l0aCB0aGUgb3JpZ2luYWwgZmxhZyB2YWx1ZVxuICAgICAgICAgICAgICAgIGlmICh0eXBlb2Ygb3JpZ2luYWxDb25maWd1cmFibGVGbGFnID09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICAgICAgICAgIGRlbGV0ZSBkZXNjLmNvbmZpZ3VyYWJsZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGRlc2MuY29uZmlndXJhYmxlID0gb3JpZ2luYWxDb25maWd1cmFibGVGbGFnO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gX2RlZmluZVByb3BlcnR5KG9iaiwgcHJvcCwgZGVzYyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGNhdGNoIChlcnJvcikge1xuICAgICAgICAgICAgICAgICAgICB2YXIgc3dhbGxvd0Vycm9yID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgICAgIGlmIChwcm9wID09PSAnY3JlYXRlZENhbGxiYWNrJyB8fCBwcm9wID09PSAnYXR0YWNoZWRDYWxsYmFjaycgfHxcbiAgICAgICAgICAgICAgICAgICAgICAgIHByb3AgPT09ICdkZXRhY2hlZENhbGxiYWNrJyB8fCBwcm9wID09PSAnYXR0cmlidXRlQ2hhbmdlZENhbGxiYWNrJykge1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gV2Ugb25seSBzd2FsbG93IHRoZSBlcnJvciBpbiByZWdpc3RlckVsZW1lbnQgcGF0Y2hcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHRoaXMgaXMgdGhlIHdvcmsgYXJvdW5kIHNpbmNlIHNvbWUgYXBwbGljYXRpb25zXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBmYWlsIGlmIHdlIHRocm93IHRoZSBlcnJvclxuICAgICAgICAgICAgICAgICAgICAgICAgc3dhbGxvd0Vycm9yID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAoIXN3YWxsb3dFcnJvcikge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgLy8gVE9ETzogQEppYUxpUGFzc2lvbiwgU29tZSBhcHBsaWNhdGlvbiBzdWNoIGFzIGByZWdpc3RlckVsZW1lbnRgIHBhdGNoXG4gICAgICAgICAgICAgICAgICAgIC8vIHN0aWxsIG5lZWQgdG8gc3dhbGxvdyB0aGUgZXJyb3IsIGluIHRoZSBmdXR1cmUgYWZ0ZXIgdGhlc2UgYXBwbGljYXRpb25zXG4gICAgICAgICAgICAgICAgICAgIC8vIGFyZSB1cGRhdGVkLCB0aGUgZm9sbG93aW5nIGxvZ2ljIGNhbiBiZSByZW1vdmVkLlxuICAgICAgICAgICAgICAgICAgICB2YXIgZGVzY0pzb24gPSBudWxsO1xuICAgICAgICAgICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICAgICAgICAgICAgZGVzY0pzb24gPSBKU09OLnN0cmluZ2lmeShkZXNjKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGRlc2NKc29uID0gZGVzYy50b1N0cmluZygpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKFwiQXR0ZW1wdGluZyB0byBjb25maWd1cmUgJ1wiLmNvbmNhdChwcm9wLCBcIicgd2l0aCBkZXNjcmlwdG9yICdcIikuY29uY2F0KGRlc2NKc29uLCBcIicgb24gb2JqZWN0ICdcIikuY29uY2F0KG9iaiwgXCInIGFuZCBnb3QgZXJyb3IsIGdpdmluZyB1cDogXCIpLmNvbmNhdChlcnJvcikpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGV2ZW50VGFyZ2V0TGVnYWN5UGF0Y2goX2dsb2JhbCwgYXBpKSB7XG4gICAgICAgIHZhciBfYiA9IGFwaS5nZXRHbG9iYWxPYmplY3RzKCksIGV2ZW50TmFtZXMgPSBfYi5ldmVudE5hbWVzLCBnbG9iYWxTb3VyY2VzID0gX2IuZ2xvYmFsU291cmNlcywgem9uZVN5bWJvbEV2ZW50TmFtZXMgPSBfYi56b25lU3ltYm9sRXZlbnROYW1lcywgVFJVRV9TVFIgPSBfYi5UUlVFX1NUUiwgRkFMU0VfU1RSID0gX2IuRkFMU0VfU1RSLCBaT05FX1NZTUJPTF9QUkVGSVggPSBfYi5aT05FX1NZTUJPTF9QUkVGSVg7XG4gICAgICAgIHZhciBXVEZfSVNTVUVfNTU1ID0gJ0FuY2hvcixBcmVhLEF1ZGlvLEJSLEJhc2UsQmFzZUZvbnQsQm9keSxCdXR0b24sQ2FudmFzLENvbnRlbnQsRExpc3QsRGlyZWN0b3J5LERpdixFbWJlZCxGaWVsZFNldCxGb250LEZvcm0sRnJhbWUsRnJhbWVTZXQsSFIsSGVhZCxIZWFkaW5nLEh0bWwsSUZyYW1lLEltYWdlLElucHV0LEtleWdlbixMSSxMYWJlbCxMZWdlbmQsTGluayxNYXAsTWFycXVlZSxNZWRpYSxNZW51LE1ldGEsTWV0ZXIsTW9kLE9MaXN0LE9iamVjdCxPcHRHcm91cCxPcHRpb24sT3V0cHV0LFBhcmFncmFwaCxQcmUsUHJvZ3Jlc3MsUXVvdGUsU2NyaXB0LFNlbGVjdCxTb3VyY2UsU3BhbixTdHlsZSxUYWJsZUNhcHRpb24sVGFibGVDZWxsLFRhYmxlQ29sLFRhYmxlLFRhYmxlUm93LFRhYmxlU2VjdGlvbixUZXh0QXJlYSxUaXRsZSxUcmFjayxVTGlzdCxVbmtub3duLFZpZGVvJztcbiAgICAgICAgdmFyIE5PX0VWRU5UX1RBUkdFVCA9ICdBcHBsaWNhdGlvbkNhY2hlLEV2ZW50U291cmNlLEZpbGVSZWFkZXIsSW5wdXRNZXRob2RDb250ZXh0LE1lZGlhQ29udHJvbGxlcixNZXNzYWdlUG9ydCxOb2RlLFBlcmZvcm1hbmNlLFNWR0VsZW1lbnRJbnN0YW5jZSxTaGFyZWRXb3JrZXIsVGV4dFRyYWNrLFRleHRUcmFja0N1ZSxUZXh0VHJhY2tMaXN0LFdlYktpdE5hbWVkRmxvdyxXaW5kb3csV29ya2VyLFdvcmtlckdsb2JhbFNjb3BlLFhNTEh0dHBSZXF1ZXN0LFhNTEh0dHBSZXF1ZXN0RXZlbnRUYXJnZXQsWE1MSHR0cFJlcXVlc3RVcGxvYWQsSURCUmVxdWVzdCxJREJPcGVuREJSZXF1ZXN0LElEQkRhdGFiYXNlLElEQlRyYW5zYWN0aW9uLElEQkN1cnNvcixEQkluZGV4LFdlYlNvY2tldCdcbiAgICAgICAgICAgIC5zcGxpdCgnLCcpO1xuICAgICAgICB2YXIgRVZFTlRfVEFSR0VUID0gJ0V2ZW50VGFyZ2V0JztcbiAgICAgICAgdmFyIGFwaXMgPSBbXTtcbiAgICAgICAgdmFyIGlzV3RmID0gX2dsb2JhbFsnd3RmJ107XG4gICAgICAgIHZhciBXVEZfSVNTVUVfNTU1X0FSUkFZID0gV1RGX0lTU1VFXzU1NS5zcGxpdCgnLCcpO1xuICAgICAgICBpZiAoaXNXdGYpIHtcbiAgICAgICAgICAgIC8vIFdvcmthcm91bmQgZm9yOiBodHRwczovL2dpdGh1Yi5jb20vZ29vZ2xlL3RyYWNpbmctZnJhbWV3b3JrL2lzc3Vlcy81NTVcbiAgICAgICAgICAgIGFwaXMgPSBXVEZfSVNTVUVfNTU1X0FSUkFZLm1hcChmdW5jdGlvbiAodikgeyByZXR1cm4gJ0hUTUwnICsgdiArICdFbGVtZW50JzsgfSkuY29uY2F0KE5PX0VWRU5UX1RBUkdFVCk7XG4gICAgICAgIH1cbiAgICAgICAgZWxzZSBpZiAoX2dsb2JhbFtFVkVOVF9UQVJHRVRdKSB7XG4gICAgICAgICAgICBhcGlzLnB1c2goRVZFTlRfVEFSR0VUKTtcbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIC8vIE5vdGU6IEV2ZW50VGFyZ2V0IGlzIG5vdCBhdmFpbGFibGUgaW4gYWxsIGJyb3dzZXJzLFxuICAgICAgICAgICAgLy8gaWYgaXQncyBub3QgYXZhaWxhYmxlLCB3ZSBpbnN0ZWFkIHBhdGNoIHRoZSBBUElzIGluIHRoZSBJREwgdGhhdCBpbmhlcml0IGZyb20gRXZlbnRUYXJnZXRcbiAgICAgICAgICAgIGFwaXMgPSBOT19FVkVOVF9UQVJHRVQ7XG4gICAgICAgIH1cbiAgICAgICAgdmFyIGlzRGlzYWJsZUlFQ2hlY2sgPSBfZ2xvYmFsWydfX1pvbmVfZGlzYWJsZV9JRV9jaGVjayddIHx8IGZhbHNlO1xuICAgICAgICB2YXIgaXNFbmFibGVDcm9zc0NvbnRleHRDaGVjayA9IF9nbG9iYWxbJ19fWm9uZV9lbmFibGVfY3Jvc3NfY29udGV4dF9jaGVjayddIHx8IGZhbHNlO1xuICAgICAgICB2YXIgaWVPckVkZ2UgPSBhcGkuaXNJRU9yRWRnZSgpO1xuICAgICAgICB2YXIgQUREX0VWRU5UX0xJU1RFTkVSX1NPVVJDRSA9ICcuYWRkRXZlbnRMaXN0ZW5lcjonO1xuICAgICAgICB2YXIgRlVOQ1RJT05fV1JBUFBFUiA9ICdbb2JqZWN0IEZ1bmN0aW9uV3JhcHBlcl0nO1xuICAgICAgICB2YXIgQlJPV1NFUl9UT09MUyA9ICdmdW5jdGlvbiBfX0JST1dTRVJUT09MU19DT05TT0xFX1NBRkVGVU5DKCkgeyBbbmF0aXZlIGNvZGVdIH0nO1xuICAgICAgICB2YXIgcG9pbnRlckV2ZW50c01hcCA9IHtcbiAgICAgICAgICAgICdNU1BvaW50ZXJDYW5jZWwnOiAncG9pbnRlcmNhbmNlbCcsXG4gICAgICAgICAgICAnTVNQb2ludGVyRG93bic6ICdwb2ludGVyZG93bicsXG4gICAgICAgICAgICAnTVNQb2ludGVyRW50ZXInOiAncG9pbnRlcmVudGVyJyxcbiAgICAgICAgICAgICdNU1BvaW50ZXJIb3Zlcic6ICdwb2ludGVyaG92ZXInLFxuICAgICAgICAgICAgJ01TUG9pbnRlckxlYXZlJzogJ3BvaW50ZXJsZWF2ZScsXG4gICAgICAgICAgICAnTVNQb2ludGVyTW92ZSc6ICdwb2ludGVybW92ZScsXG4gICAgICAgICAgICAnTVNQb2ludGVyT3V0JzogJ3BvaW50ZXJvdXQnLFxuICAgICAgICAgICAgJ01TUG9pbnRlck92ZXInOiAncG9pbnRlcm92ZXInLFxuICAgICAgICAgICAgJ01TUG9pbnRlclVwJzogJ3BvaW50ZXJ1cCdcbiAgICAgICAgfTtcbiAgICAgICAgLy8gIHByZWRlZmluZSBhbGwgX196b25lX3N5bWJvbF9fICsgZXZlbnROYW1lICsgdHJ1ZS9mYWxzZSBzdHJpbmdcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBldmVudE5hbWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICB2YXIgZXZlbnROYW1lID0gZXZlbnROYW1lc1tpXTtcbiAgICAgICAgICAgIHZhciBmYWxzZUV2ZW50TmFtZSA9IGV2ZW50TmFtZSArIEZBTFNFX1NUUjtcbiAgICAgICAgICAgIHZhciB0cnVlRXZlbnROYW1lID0gZXZlbnROYW1lICsgVFJVRV9TVFI7XG4gICAgICAgICAgICB2YXIgc3ltYm9sID0gWk9ORV9TWU1CT0xfUFJFRklYICsgZmFsc2VFdmVudE5hbWU7XG4gICAgICAgICAgICB2YXIgc3ltYm9sQ2FwdHVyZSA9IFpPTkVfU1lNQk9MX1BSRUZJWCArIHRydWVFdmVudE5hbWU7XG4gICAgICAgICAgICB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdID0ge307XG4gICAgICAgICAgICB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdW0ZBTFNFX1NUUl0gPSBzeW1ib2w7XG4gICAgICAgICAgICB6b25lU3ltYm9sRXZlbnROYW1lc1tldmVudE5hbWVdW1RSVUVfU1RSXSA9IHN5bWJvbENhcHR1cmU7XG4gICAgICAgIH1cbiAgICAgICAgLy8gIHByZWRlZmluZSBhbGwgdGFzay5zb3VyY2Ugc3RyaW5nXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgV1RGX0lTU1VFXzU1NV9BUlJBWS5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgdmFyIHRhcmdldCA9IFdURl9JU1NVRV81NTVfQVJSQVlbaV07XG4gICAgICAgICAgICB2YXIgdGFyZ2V0cyA9IGdsb2JhbFNvdXJjZXNbdGFyZ2V0XSA9IHt9O1xuICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBldmVudE5hbWVzLmxlbmd0aDsgaisrKSB7XG4gICAgICAgICAgICAgICAgdmFyIGV2ZW50TmFtZSA9IGV2ZW50TmFtZXNbal07XG4gICAgICAgICAgICAgICAgdGFyZ2V0c1tldmVudE5hbWVdID0gdGFyZ2V0ICsgQUREX0VWRU5UX0xJU1RFTkVSX1NPVVJDRSArIGV2ZW50TmFtZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICB2YXIgY2hlY2tJRUFuZENyb3NzQ29udGV4dCA9IGZ1bmN0aW9uIChuYXRpdmVEZWxlZ2F0ZSwgZGVsZWdhdGUsIHRhcmdldCwgYXJncykge1xuICAgICAgICAgICAgaWYgKCFpc0Rpc2FibGVJRUNoZWNrICYmIGllT3JFZGdlKSB7XG4gICAgICAgICAgICAgICAgaWYgKGlzRW5hYmxlQ3Jvc3NDb250ZXh0Q2hlY2spIHtcbiAgICAgICAgICAgICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0ZXN0U3RyaW5nID0gZGVsZWdhdGUudG9TdHJpbmcoKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmICgodGVzdFN0cmluZyA9PT0gRlVOQ1RJT05fV1JBUFBFUiB8fCB0ZXN0U3RyaW5nID09IEJST1dTRVJfVE9PTFMpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbmF0aXZlRGVsZWdhdGUuYXBwbHkodGFyZ2V0LCBhcmdzKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBuYXRpdmVEZWxlZ2F0ZS5hcHBseSh0YXJnZXQsIGFyZ3MpO1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICB2YXIgdGVzdFN0cmluZyA9IGRlbGVnYXRlLnRvU3RyaW5nKCk7XG4gICAgICAgICAgICAgICAgICAgIGlmICgodGVzdFN0cmluZyA9PT0gRlVOQ1RJT05fV1JBUFBFUiB8fCB0ZXN0U3RyaW5nID09IEJST1dTRVJfVE9PTFMpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBuYXRpdmVEZWxlZ2F0ZS5hcHBseSh0YXJnZXQsIGFyZ3MpO1xuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZWxzZSBpZiAoaXNFbmFibGVDcm9zc0NvbnRleHRDaGVjaykge1xuICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgIGRlbGVnYXRlLnRvU3RyaW5nKCk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGNhdGNoIChlcnJvcikge1xuICAgICAgICAgICAgICAgICAgICBuYXRpdmVEZWxlZ2F0ZS5hcHBseSh0YXJnZXQsIGFyZ3MpO1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgIH07XG4gICAgICAgIHZhciBhcGlUeXBlcyA9IFtdO1xuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGFwaXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgIHZhciB0eXBlID0gX2dsb2JhbFthcGlzW2ldXTtcbiAgICAgICAgICAgIGFwaVR5cGVzLnB1c2godHlwZSAmJiB0eXBlLnByb3RvdHlwZSk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gdmggaXMgdmFsaWRhdGVIYW5kbGVyIHRvIGNoZWNrIGV2ZW50IGhhbmRsZXJcbiAgICAgICAgLy8gaXMgdmFsaWQgb3Igbm90KGZvciBzZWN1cml0eSBjaGVjaylcbiAgICAgICAgYXBpLnBhdGNoRXZlbnRUYXJnZXQoX2dsb2JhbCwgYXBpLCBhcGlUeXBlcywge1xuICAgICAgICAgICAgdmg6IGNoZWNrSUVBbmRDcm9zc0NvbnRleHQsXG4gICAgICAgICAgICB0cmFuc2ZlckV2ZW50TmFtZTogZnVuY3Rpb24gKGV2ZW50TmFtZSkge1xuICAgICAgICAgICAgICAgIHZhciBwb2ludGVyRXZlbnROYW1lID0gcG9pbnRlckV2ZW50c01hcFtldmVudE5hbWVdO1xuICAgICAgICAgICAgICAgIHJldHVybiBwb2ludGVyRXZlbnROYW1lIHx8IGV2ZW50TmFtZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIFpvbmVbYXBpLnN5bWJvbCgncGF0Y2hFdmVudFRhcmdldCcpXSA9ICEhX2dsb2JhbFtFVkVOVF9UQVJHRVRdO1xuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQGxpY2Vuc2VcbiAgICAgKiBDb3B5cmlnaHQgR29vZ2xlIExMQyBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICAgICAqXG4gICAgICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAgICAgKiBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIGF0IGh0dHBzOi8vYW5ndWxhci5pby9saWNlbnNlXG4gICAgICovXG4gICAgLy8gd2UgaGF2ZSB0byBwYXRjaCB0aGUgaW5zdGFuY2Ugc2luY2UgdGhlIHByb3RvIGlzIG5vbi1jb25maWd1cmFibGVcbiAgICBmdW5jdGlvbiBhcHBseShhcGksIF9nbG9iYWwpIHtcbiAgICAgICAgdmFyIF9iID0gYXBpLmdldEdsb2JhbE9iamVjdHMoKSwgQUREX0VWRU5UX0xJU1RFTkVSX1NUUiA9IF9iLkFERF9FVkVOVF9MSVNURU5FUl9TVFIsIFJFTU9WRV9FVkVOVF9MSVNURU5FUl9TVFIgPSBfYi5SRU1PVkVfRVZFTlRfTElTVEVORVJfU1RSO1xuICAgICAgICB2YXIgV1MgPSBfZ2xvYmFsLldlYlNvY2tldDtcbiAgICAgICAgLy8gT24gU2FmYXJpIHdpbmRvdy5FdmVudFRhcmdldCBkb2Vzbid0IGV4aXN0IHNvIG5lZWQgdG8gcGF0Y2ggV1MgYWRkL3JlbW92ZUV2ZW50TGlzdGVuZXJcbiAgICAgICAgLy8gT24gb2xkZXIgQ2hyb21lLCBubyBuZWVkIHNpbmNlIEV2ZW50VGFyZ2V0IHdhcyBhbHJlYWR5IHBhdGNoZWRcbiAgICAgICAgaWYgKCFfZ2xvYmFsLkV2ZW50VGFyZ2V0KSB7XG4gICAgICAgICAgICBhcGkucGF0Y2hFdmVudFRhcmdldChfZ2xvYmFsLCBhcGksIFtXUy5wcm90b3R5cGVdKTtcbiAgICAgICAgfVxuICAgICAgICBfZ2xvYmFsLldlYlNvY2tldCA9IGZ1bmN0aW9uICh4LCB5KSB7XG4gICAgICAgICAgICB2YXIgc29ja2V0ID0gYXJndW1lbnRzLmxlbmd0aCA+IDEgPyBuZXcgV1MoeCwgeSkgOiBuZXcgV1MoeCk7XG4gICAgICAgICAgICB2YXIgcHJveHlTb2NrZXQ7XG4gICAgICAgICAgICB2YXIgcHJveHlTb2NrZXRQcm90bztcbiAgICAgICAgICAgIC8vIFNhZmFyaSA3LjAgaGFzIG5vbi1jb25maWd1cmFibGUgb3duICdvbm1lc3NhZ2UnIGFuZCBmcmllbmRzIHByb3BlcnRpZXMgb24gdGhlIHNvY2tldCBpbnN0YW5jZVxuICAgICAgICAgICAgdmFyIG9ubWVzc2FnZURlc2MgPSBhcGkuT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHNvY2tldCwgJ29ubWVzc2FnZScpO1xuICAgICAgICAgICAgaWYgKG9ubWVzc2FnZURlc2MgJiYgb25tZXNzYWdlRGVzYy5jb25maWd1cmFibGUgPT09IGZhbHNlKSB7XG4gICAgICAgICAgICAgICAgcHJveHlTb2NrZXQgPSBhcGkuT2JqZWN0Q3JlYXRlKHNvY2tldCk7XG4gICAgICAgICAgICAgICAgLy8gc29ja2V0IGhhdmUgb3duIHByb3BlcnR5IGRlc2NyaXB0b3IgJ29ub3BlbicsICdvbm1lc3NhZ2UnLCAnb25jbG9zZScsICdvbmVycm9yJ1xuICAgICAgICAgICAgICAgIC8vIGJ1dCBwcm94eVNvY2tldCBub3QsIHNvIHdlIHdpbGwga2VlcCBzb2NrZXQgYXMgcHJvdG90eXBlIGFuZCBwYXNzIGl0IHRvXG4gICAgICAgICAgICAgICAgLy8gcGF0Y2hPblByb3BlcnRpZXMgbWV0aG9kXG4gICAgICAgICAgICAgICAgcHJveHlTb2NrZXRQcm90byA9IHNvY2tldDtcbiAgICAgICAgICAgICAgICBbQUREX0VWRU5UX0xJU1RFTkVSX1NUUiwgUkVNT1ZFX0VWRU5UX0xJU1RFTkVSX1NUUiwgJ3NlbmQnLCAnY2xvc2UnXS5mb3JFYWNoKGZ1bmN0aW9uIChwcm9wTmFtZSkge1xuICAgICAgICAgICAgICAgICAgICBwcm94eVNvY2tldFtwcm9wTmFtZV0gPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgYXJncyA9IGFwaS5BcnJheVNsaWNlLmNhbGwoYXJndW1lbnRzKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChwcm9wTmFtZSA9PT0gQUREX0VWRU5UX0xJU1RFTkVSX1NUUiB8fCBwcm9wTmFtZSA9PT0gUkVNT1ZFX0VWRU5UX0xJU1RFTkVSX1NUUikge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBldmVudE5hbWUgPSBhcmdzLmxlbmd0aCA+IDAgPyBhcmdzWzBdIDogdW5kZWZpbmVkO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChldmVudE5hbWUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHByb3BlcnR5U3ltYm9sID0gWm9uZS5fX3N5bWJvbF9fKCdPTl9QUk9QRVJUWScgKyBldmVudE5hbWUpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzb2NrZXRbcHJvcGVydHlTeW1ib2xdID0gcHJveHlTb2NrZXRbcHJvcGVydHlTeW1ib2xdO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBzb2NrZXRbcHJvcE5hbWVdLmFwcGx5KHNvY2tldCwgYXJncyk7XG4gICAgICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAvLyB3ZSBjYW4gcGF0Y2ggdGhlIHJlYWwgc29ja2V0XG4gICAgICAgICAgICAgICAgcHJveHlTb2NrZXQgPSBzb2NrZXQ7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBhcGkucGF0Y2hPblByb3BlcnRpZXMocHJveHlTb2NrZXQsIFsnY2xvc2UnLCAnZXJyb3InLCAnbWVzc2FnZScsICdvcGVuJ10sIHByb3h5U29ja2V0UHJvdG8pO1xuICAgICAgICAgICAgcmV0dXJuIHByb3h5U29ja2V0O1xuICAgICAgICB9O1xuICAgICAgICB2YXIgZ2xvYmFsV2ViU29ja2V0ID0gX2dsb2JhbFsnV2ViU29ja2V0J107XG4gICAgICAgIGZvciAodmFyIHByb3AgaW4gV1MpIHtcbiAgICAgICAgICAgIGdsb2JhbFdlYlNvY2tldFtwcm9wXSA9IFdTW3Byb3BdO1xuICAgICAgICB9XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHByb3BlcnR5RGVzY3JpcHRvckxlZ2FjeVBhdGNoKGFwaSwgX2dsb2JhbCkge1xuICAgICAgICB2YXIgX2IgPSBhcGkuZ2V0R2xvYmFsT2JqZWN0cygpLCBpc05vZGUgPSBfYi5pc05vZGUsIGlzTWl4ID0gX2IuaXNNaXg7XG4gICAgICAgIGlmIChpc05vZGUgJiYgIWlzTWl4KSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgaWYgKCFjYW5QYXRjaFZpYVByb3BlcnR5RGVzY3JpcHRvcihhcGksIF9nbG9iYWwpKSB7XG4gICAgICAgICAgICB2YXIgc3VwcG9ydHNXZWJTb2NrZXQgPSB0eXBlb2YgV2ViU29ja2V0ICE9PSAndW5kZWZpbmVkJztcbiAgICAgICAgICAgIC8vIFNhZmFyaSwgQW5kcm9pZCBicm93c2VycyAoSmVsbHkgQmVhbilcbiAgICAgICAgICAgIHBhdGNoVmlhQ2FwdHVyaW5nQWxsVGhlRXZlbnRzKGFwaSk7XG4gICAgICAgICAgICBhcGkucGF0Y2hDbGFzcygnWE1MSHR0cFJlcXVlc3QnKTtcbiAgICAgICAgICAgIGlmIChzdXBwb3J0c1dlYlNvY2tldCkge1xuICAgICAgICAgICAgICAgIGFwcGx5KGFwaSwgX2dsb2JhbCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBab25lW2FwaS5zeW1ib2woJ3BhdGNoRXZlbnRzJyldID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgIH1cbiAgICBmdW5jdGlvbiBjYW5QYXRjaFZpYVByb3BlcnR5RGVzY3JpcHRvcihhcGksIF9nbG9iYWwpIHtcbiAgICAgICAgdmFyIF9iID0gYXBpLmdldEdsb2JhbE9iamVjdHMoKSwgaXNCcm93c2VyID0gX2IuaXNCcm93c2VyLCBpc01peCA9IF9iLmlzTWl4O1xuICAgICAgICBpZiAoKGlzQnJvd3NlciB8fCBpc01peCkgJiZcbiAgICAgICAgICAgICFhcGkuT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKEhUTUxFbGVtZW50LnByb3RvdHlwZSwgJ29uY2xpY2snKSAmJlxuICAgICAgICAgICAgdHlwZW9mIEVsZW1lbnQgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICAvLyBXZWJLaXQgaHR0cHM6Ly9idWdzLndlYmtpdC5vcmcvc2hvd19idWcuY2dpP2lkPTEzNDM2NFxuICAgICAgICAgICAgLy8gSURMIGludGVyZmFjZSBhdHRyaWJ1dGVzIGFyZSBub3QgY29uZmlndXJhYmxlXG4gICAgICAgICAgICB2YXIgZGVzYyA9IGFwaS5PYmplY3RHZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3IoRWxlbWVudC5wcm90b3R5cGUsICdvbmNsaWNrJyk7XG4gICAgICAgICAgICBpZiAoZGVzYyAmJiAhZGVzYy5jb25maWd1cmFibGUpXG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgLy8gdHJ5IHRvIHVzZSBvbmNsaWNrIHRvIGRldGVjdCB3aGV0aGVyIHdlIGNhbiBwYXRjaCB2aWEgcHJvcGVydHlEZXNjcmlwdG9yXG4gICAgICAgICAgICAvLyBiZWNhdXNlIFhNTEh0dHBSZXF1ZXN0IGlzIG5vdCBhdmFpbGFibGUgaW4gc2VydmljZSB3b3JrZXJcbiAgICAgICAgICAgIGlmIChkZXNjKSB7XG4gICAgICAgICAgICAgICAgYXBpLk9iamVjdERlZmluZVByb3BlcnR5KEVsZW1lbnQucHJvdG90eXBlLCAnb25jbGljaycsIHtcbiAgICAgICAgICAgICAgICAgICAgZW51bWVyYWJsZTogdHJ1ZSxcbiAgICAgICAgICAgICAgICAgICAgY29uZmlndXJhYmxlOiB0cnVlLFxuICAgICAgICAgICAgICAgICAgICBnZXQ6IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgdmFyIGRpdiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuICAgICAgICAgICAgICAgIHZhciByZXN1bHQgPSAhIWRpdi5vbmNsaWNrO1xuICAgICAgICAgICAgICAgIGFwaS5PYmplY3REZWZpbmVQcm9wZXJ0eShFbGVtZW50LnByb3RvdHlwZSwgJ29uY2xpY2snLCBkZXNjKTtcbiAgICAgICAgICAgICAgICByZXR1cm4gcmVzdWx0O1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHZhciBYTUxIdHRwUmVxdWVzdCA9IF9nbG9iYWxbJ1hNTEh0dHBSZXF1ZXN0J107XG4gICAgICAgIGlmICghWE1MSHR0cFJlcXVlc3QpIHtcbiAgICAgICAgICAgIC8vIFhNTEh0dHBSZXF1ZXN0IGlzIG5vdCBhdmFpbGFibGUgaW4gc2VydmljZSB3b3JrZXJcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgfVxuICAgICAgICB2YXIgT05fUkVBRFlfU1RBVEVfQ0hBTkdFID0gJ29ucmVhZHlzdGF0ZWNoYW5nZSc7XG4gICAgICAgIHZhciBYTUxIdHRwUmVxdWVzdFByb3RvdHlwZSA9IFhNTEh0dHBSZXF1ZXN0LnByb3RvdHlwZTtcbiAgICAgICAgdmFyIHhockRlc2MgPSBhcGkuT2JqZWN0R2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKFhNTEh0dHBSZXF1ZXN0UHJvdG90eXBlLCBPTl9SRUFEWV9TVEFURV9DSEFOR0UpO1xuICAgICAgICAvLyBhZGQgZW51bWVyYWJsZSBhbmQgY29uZmlndXJhYmxlIGhlcmUgYmVjYXVzZSBpbiBvcGVyYVxuICAgICAgICAvLyBieSBkZWZhdWx0IFhNTEh0dHBSZXF1ZXN0LnByb3RvdHlwZS5vbnJlYWR5c3RhdGVjaGFuZ2UgaXMgdW5kZWZpbmVkXG4gICAgICAgIC8vIHdpdGhvdXQgYWRkaW5nIGVudW1lcmFibGUgYW5kIGNvbmZpZ3VyYWJsZSB3aWxsIGNhdXNlIG9ucmVhZHlzdGF0ZWNoYW5nZVxuICAgICAgICAvLyBub24tY29uZmlndXJhYmxlXG4gICAgICAgIC8vIGFuZCBpZiBYTUxIdHRwUmVxdWVzdC5wcm90b3R5cGUub25yZWFkeXN0YXRlY2hhbmdlIGlzIHVuZGVmaW5lZCxcbiAgICAgICAgLy8gd2Ugc2hvdWxkIHNldCBhIHJlYWwgZGVzYyBpbnN0ZWFkIGEgZmFrZSBvbmVcbiAgICAgICAgaWYgKHhockRlc2MpIHtcbiAgICAgICAgICAgIGFwaS5PYmplY3REZWZpbmVQcm9wZXJ0eShYTUxIdHRwUmVxdWVzdFByb3RvdHlwZSwgT05fUkVBRFlfU1RBVEVfQ0hBTkdFLCB7XG4gICAgICAgICAgICAgICAgZW51bWVyYWJsZTogdHJ1ZSxcbiAgICAgICAgICAgICAgICBjb25maWd1cmFibGU6IHRydWUsXG4gICAgICAgICAgICAgICAgZ2V0OiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgdmFyIHJlcSA9IG5ldyBYTUxIdHRwUmVxdWVzdCgpO1xuICAgICAgICAgICAgdmFyIHJlc3VsdCA9ICEhcmVxLm9ucmVhZHlzdGF0ZWNoYW5nZTtcbiAgICAgICAgICAgIC8vIHJlc3RvcmUgb3JpZ2luYWwgZGVzY1xuICAgICAgICAgICAgYXBpLk9iamVjdERlZmluZVByb3BlcnR5KFhNTEh0dHBSZXF1ZXN0UHJvdG90eXBlLCBPTl9SRUFEWV9TVEFURV9DSEFOR0UsIHhockRlc2MgfHwge30pO1xuICAgICAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIHZhciBTWU1CT0xfRkFLRV9PTlJFQURZU1RBVEVDSEFOR0VfMSA9IGFwaS5zeW1ib2woJ2Zha2UnKTtcbiAgICAgICAgICAgIGFwaS5PYmplY3REZWZpbmVQcm9wZXJ0eShYTUxIdHRwUmVxdWVzdFByb3RvdHlwZSwgT05fUkVBRFlfU1RBVEVfQ0hBTkdFLCB7XG4gICAgICAgICAgICAgICAgZW51bWVyYWJsZTogdHJ1ZSxcbiAgICAgICAgICAgICAgICBjb25maWd1cmFibGU6IHRydWUsXG4gICAgICAgICAgICAgICAgZ2V0OiBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0aGlzW1NZTUJPTF9GQUtFX09OUkVBRFlTVEFURUNIQU5HRV8xXTtcbiAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgIHNldDogZnVuY3Rpb24gKHZhbHVlKSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXNbU1lNQk9MX0ZBS0VfT05SRUFEWVNUQVRFQ0hBTkdFXzFdID0gdmFsdWU7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB2YXIgcmVxID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG4gICAgICAgICAgICB2YXIgZGV0ZWN0RnVuYyA9IGZ1bmN0aW9uICgpIHsgfTtcbiAgICAgICAgICAgIHJlcS5vbnJlYWR5c3RhdGVjaGFuZ2UgPSBkZXRlY3RGdW5jO1xuICAgICAgICAgICAgdmFyIHJlc3VsdCA9IHJlcVtTWU1CT0xfRkFLRV9PTlJFQURZU1RBVEVDSEFOR0VfMV0gPT09IGRldGVjdEZ1bmM7XG4gICAgICAgICAgICByZXEub25yZWFkeXN0YXRlY2hhbmdlID0gbnVsbDtcbiAgICAgICAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgICAgIH1cbiAgICB9XG4gICAgdmFyIGdsb2JhbEV2ZW50SGFuZGxlcnNFdmVudE5hbWVzID0gW1xuICAgICAgICAnYWJvcnQnLFxuICAgICAgICAnYW5pbWF0aW9uY2FuY2VsJyxcbiAgICAgICAgJ2FuaW1hdGlvbmVuZCcsXG4gICAgICAgICdhbmltYXRpb25pdGVyYXRpb24nLFxuICAgICAgICAnYXV4Y2xpY2snLFxuICAgICAgICAnYmVmb3JlaW5wdXQnLFxuICAgICAgICAnYmx1cicsXG4gICAgICAgICdjYW5jZWwnLFxuICAgICAgICAnY2FucGxheScsXG4gICAgICAgICdjYW5wbGF5dGhyb3VnaCcsXG4gICAgICAgICdjaGFuZ2UnLFxuICAgICAgICAnY29tcG9zaXRpb25zdGFydCcsXG4gICAgICAgICdjb21wb3NpdGlvbnVwZGF0ZScsXG4gICAgICAgICdjb21wb3NpdGlvbmVuZCcsXG4gICAgICAgICdjdWVjaGFuZ2UnLFxuICAgICAgICAnY2xpY2snLFxuICAgICAgICAnY2xvc2UnLFxuICAgICAgICAnY29udGV4dG1lbnUnLFxuICAgICAgICAnY3VyZWNoYW5nZScsXG4gICAgICAgICdkYmxjbGljaycsXG4gICAgICAgICdkcmFnJyxcbiAgICAgICAgJ2RyYWdlbmQnLFxuICAgICAgICAnZHJhZ2VudGVyJyxcbiAgICAgICAgJ2RyYWdleGl0JyxcbiAgICAgICAgJ2RyYWdsZWF2ZScsXG4gICAgICAgICdkcmFnb3ZlcicsXG4gICAgICAgICdkcm9wJyxcbiAgICAgICAgJ2R1cmF0aW9uY2hhbmdlJyxcbiAgICAgICAgJ2VtcHRpZWQnLFxuICAgICAgICAnZW5kZWQnLFxuICAgICAgICAnZXJyb3InLFxuICAgICAgICAnZm9jdXMnLFxuICAgICAgICAnZm9jdXNpbicsXG4gICAgICAgICdmb2N1c291dCcsXG4gICAgICAgICdnb3Rwb2ludGVyY2FwdHVyZScsXG4gICAgICAgICdpbnB1dCcsXG4gICAgICAgICdpbnZhbGlkJyxcbiAgICAgICAgJ2tleWRvd24nLFxuICAgICAgICAna2V5cHJlc3MnLFxuICAgICAgICAna2V5dXAnLFxuICAgICAgICAnbG9hZCcsXG4gICAgICAgICdsb2Fkc3RhcnQnLFxuICAgICAgICAnbG9hZGVkZGF0YScsXG4gICAgICAgICdsb2FkZWRtZXRhZGF0YScsXG4gICAgICAgICdsb3N0cG9pbnRlcmNhcHR1cmUnLFxuICAgICAgICAnbW91c2Vkb3duJyxcbiAgICAgICAgJ21vdXNlZW50ZXInLFxuICAgICAgICAnbW91c2VsZWF2ZScsXG4gICAgICAgICdtb3VzZW1vdmUnLFxuICAgICAgICAnbW91c2VvdXQnLFxuICAgICAgICAnbW91c2VvdmVyJyxcbiAgICAgICAgJ21vdXNldXAnLFxuICAgICAgICAnbW91c2V3aGVlbCcsXG4gICAgICAgICdvcmllbnRhdGlvbmNoYW5nZScsXG4gICAgICAgICdwYXVzZScsXG4gICAgICAgICdwbGF5JyxcbiAgICAgICAgJ3BsYXlpbmcnLFxuICAgICAgICAncG9pbnRlcmNhbmNlbCcsXG4gICAgICAgICdwb2ludGVyZG93bicsXG4gICAgICAgICdwb2ludGVyZW50ZXInLFxuICAgICAgICAncG9pbnRlcmxlYXZlJyxcbiAgICAgICAgJ3BvaW50ZXJsb2NrY2hhbmdlJyxcbiAgICAgICAgJ21venBvaW50ZXJsb2NrY2hhbmdlJyxcbiAgICAgICAgJ3dlYmtpdHBvaW50ZXJsb2NrZXJjaGFuZ2UnLFxuICAgICAgICAncG9pbnRlcmxvY2tlcnJvcicsXG4gICAgICAgICdtb3pwb2ludGVybG9ja2Vycm9yJyxcbiAgICAgICAgJ3dlYmtpdHBvaW50ZXJsb2NrZXJyb3InLFxuICAgICAgICAncG9pbnRlcm1vdmUnLFxuICAgICAgICAncG9pbnRvdXQnLFxuICAgICAgICAncG9pbnRlcm92ZXInLFxuICAgICAgICAncG9pbnRlcnVwJyxcbiAgICAgICAgJ3Byb2dyZXNzJyxcbiAgICAgICAgJ3JhdGVjaGFuZ2UnLFxuICAgICAgICAncmVzZXQnLFxuICAgICAgICAncmVzaXplJyxcbiAgICAgICAgJ3Njcm9sbCcsXG4gICAgICAgICdzZWVrZWQnLFxuICAgICAgICAnc2Vla2luZycsXG4gICAgICAgICdzZWxlY3QnLFxuICAgICAgICAnc2VsZWN0aW9uY2hhbmdlJyxcbiAgICAgICAgJ3NlbGVjdHN0YXJ0JyxcbiAgICAgICAgJ3Nob3cnLFxuICAgICAgICAnc29ydCcsXG4gICAgICAgICdzdGFsbGVkJyxcbiAgICAgICAgJ3N1Ym1pdCcsXG4gICAgICAgICdzdXNwZW5kJyxcbiAgICAgICAgJ3RpbWV1cGRhdGUnLFxuICAgICAgICAndm9sdW1lY2hhbmdlJyxcbiAgICAgICAgJ3RvdWNoY2FuY2VsJyxcbiAgICAgICAgJ3RvdWNobW92ZScsXG4gICAgICAgICd0b3VjaHN0YXJ0JyxcbiAgICAgICAgJ3RvdWNoZW5kJyxcbiAgICAgICAgJ3RyYW5zaXRpb25jYW5jZWwnLFxuICAgICAgICAndHJhbnNpdGlvbmVuZCcsXG4gICAgICAgICd3YWl0aW5nJyxcbiAgICAgICAgJ3doZWVsJ1xuICAgIF07XG4gICAgdmFyIGRvY3VtZW50RXZlbnROYW1lcyA9IFtcbiAgICAgICAgJ2FmdGVyc2NyaXB0ZXhlY3V0ZScsICdiZWZvcmVzY3JpcHRleGVjdXRlJywgJ0RPTUNvbnRlbnRMb2FkZWQnLCAnZnJlZXplJywgJ2Z1bGxzY3JlZW5jaGFuZ2UnLFxuICAgICAgICAnbW96ZnVsbHNjcmVlbmNoYW5nZScsICd3ZWJraXRmdWxsc2NyZWVuY2hhbmdlJywgJ21zZnVsbHNjcmVlbmNoYW5nZScsICdmdWxsc2NyZWVuZXJyb3InLFxuICAgICAgICAnbW96ZnVsbHNjcmVlbmVycm9yJywgJ3dlYmtpdGZ1bGxzY3JlZW5lcnJvcicsICdtc2Z1bGxzY3JlZW5lcnJvcicsICdyZWFkeXN0YXRlY2hhbmdlJyxcbiAgICAgICAgJ3Zpc2liaWxpdHljaGFuZ2UnLCAncmVzdW1lJ1xuICAgIF07XG4gICAgdmFyIHdpbmRvd0V2ZW50TmFtZXMgPSBbXG4gICAgICAgICdhYnNvbHV0ZWRldmljZW9yaWVudGF0aW9uJyxcbiAgICAgICAgJ2FmdGVyaW5wdXQnLFxuICAgICAgICAnYWZ0ZXJwcmludCcsXG4gICAgICAgICdhcHBpbnN0YWxsZWQnLFxuICAgICAgICAnYmVmb3JlaW5zdGFsbHByb21wdCcsXG4gICAgICAgICdiZWZvcmVwcmludCcsXG4gICAgICAgICdiZWZvcmV1bmxvYWQnLFxuICAgICAgICAnZGV2aWNlbGlnaHQnLFxuICAgICAgICAnZGV2aWNlbW90aW9uJyxcbiAgICAgICAgJ2RldmljZW9yaWVudGF0aW9uJyxcbiAgICAgICAgJ2RldmljZW9yaWVudGF0aW9uYWJzb2x1dGUnLFxuICAgICAgICAnZGV2aWNlcHJveGltaXR5JyxcbiAgICAgICAgJ2hhc2hjaGFuZ2UnLFxuICAgICAgICAnbGFuZ3VhZ2VjaGFuZ2UnLFxuICAgICAgICAnbWVzc2FnZScsXG4gICAgICAgICdtb3piZWZvcmVwYWludCcsXG4gICAgICAgICdvZmZsaW5lJyxcbiAgICAgICAgJ29ubGluZScsXG4gICAgICAgICdwYWludCcsXG4gICAgICAgICdwYWdlc2hvdycsXG4gICAgICAgICdwYWdlaGlkZScsXG4gICAgICAgICdwb3BzdGF0ZScsXG4gICAgICAgICdyZWplY3Rpb25oYW5kbGVkJyxcbiAgICAgICAgJ3N0b3JhZ2UnLFxuICAgICAgICAndW5oYW5kbGVkcmVqZWN0aW9uJyxcbiAgICAgICAgJ3VubG9hZCcsXG4gICAgICAgICd1c2VycHJveGltaXR5JyxcbiAgICAgICAgJ3ZyZGlzcGxheWNvbm5lY3RlZCcsXG4gICAgICAgICd2cmRpc3BsYXlkaXNjb25uZWN0ZWQnLFxuICAgICAgICAndnJkaXNwbGF5cHJlc2VudGNoYW5nZSdcbiAgICBdO1xuICAgIHZhciBodG1sRWxlbWVudEV2ZW50TmFtZXMgPSBbXG4gICAgICAgICdiZWZvcmVjb3B5JywgJ2JlZm9yZWN1dCcsICdiZWZvcmVwYXN0ZScsICdjb3B5JywgJ2N1dCcsICdwYXN0ZScsICdkcmFnc3RhcnQnLCAnbG9hZGVuZCcsXG4gICAgICAgICdhbmltYXRpb25zdGFydCcsICdzZWFyY2gnLCAndHJhbnNpdGlvbnJ1bicsICd0cmFuc2l0aW9uc3RhcnQnLCAnd2Via2l0YW5pbWF0aW9uZW5kJyxcbiAgICAgICAgJ3dlYmtpdGFuaW1hdGlvbml0ZXJhdGlvbicsICd3ZWJraXRhbmltYXRpb25zdGFydCcsICd3ZWJraXR0cmFuc2l0aW9uZW5kJ1xuICAgIF07XG4gICAgdmFyIGllRWxlbWVudEV2ZW50TmFtZXMgPSBbXG4gICAgICAgICdhY3RpdmF0ZScsXG4gICAgICAgICdhZnRlcnVwZGF0ZScsXG4gICAgICAgICdhcmlhcmVxdWVzdCcsXG4gICAgICAgICdiZWZvcmVhY3RpdmF0ZScsXG4gICAgICAgICdiZWZvcmVkZWFjdGl2YXRlJyxcbiAgICAgICAgJ2JlZm9yZWVkaXRmb2N1cycsXG4gICAgICAgICdiZWZvcmV1cGRhdGUnLFxuICAgICAgICAnY2VsbGNoYW5nZScsXG4gICAgICAgICdjb250cm9sc2VsZWN0JyxcbiAgICAgICAgJ2RhdGFhdmFpbGFibGUnLFxuICAgICAgICAnZGF0YXNldGNoYW5nZWQnLFxuICAgICAgICAnZGF0YXNldGNvbXBsZXRlJyxcbiAgICAgICAgJ2Vycm9ydXBkYXRlJyxcbiAgICAgICAgJ2ZpbHRlcmNoYW5nZScsXG4gICAgICAgICdsYXlvdXRjb21wbGV0ZScsXG4gICAgICAgICdsb3NlY2FwdHVyZScsXG4gICAgICAgICdtb3ZlJyxcbiAgICAgICAgJ21vdmVlbmQnLFxuICAgICAgICAnbW92ZXN0YXJ0JyxcbiAgICAgICAgJ3Byb3BlcnR5Y2hhbmdlJyxcbiAgICAgICAgJ3Jlc2l6ZWVuZCcsXG4gICAgICAgICdyZXNpemVzdGFydCcsXG4gICAgICAgICdyb3dlbnRlcicsXG4gICAgICAgICdyb3dleGl0JyxcbiAgICAgICAgJ3Jvd3NkZWxldGUnLFxuICAgICAgICAncm93c2luc2VydGVkJyxcbiAgICAgICAgJ2NvbW1hbmQnLFxuICAgICAgICAnY29tcGFzc25lZWRzY2FsaWJyYXRpb24nLFxuICAgICAgICAnZGVhY3RpdmF0ZScsXG4gICAgICAgICdoZWxwJyxcbiAgICAgICAgJ21zY29udGVudHpvb20nLFxuICAgICAgICAnbXNtYW5pcHVsYXRpb25zdGF0ZWNoYW5nZWQnLFxuICAgICAgICAnbXNnZXN0dXJlY2hhbmdlJyxcbiAgICAgICAgJ21zZ2VzdHVyZWRvdWJsZXRhcCcsXG4gICAgICAgICdtc2dlc3R1cmVlbmQnLFxuICAgICAgICAnbXNnZXN0dXJlaG9sZCcsXG4gICAgICAgICdtc2dlc3R1cmVzdGFydCcsXG4gICAgICAgICdtc2dlc3R1cmV0YXAnLFxuICAgICAgICAnbXNnb3Rwb2ludGVyY2FwdHVyZScsXG4gICAgICAgICdtc2luZXJ0aWFzdGFydCcsXG4gICAgICAgICdtc2xvc3Rwb2ludGVyY2FwdHVyZScsXG4gICAgICAgICdtc3BvaW50ZXJjYW5jZWwnLFxuICAgICAgICAnbXNwb2ludGVyZG93bicsXG4gICAgICAgICdtc3BvaW50ZXJlbnRlcicsXG4gICAgICAgICdtc3BvaW50ZXJob3ZlcicsXG4gICAgICAgICdtc3BvaW50ZXJsZWF2ZScsXG4gICAgICAgICdtc3BvaW50ZXJtb3ZlJyxcbiAgICAgICAgJ21zcG9pbnRlcm91dCcsXG4gICAgICAgICdtc3BvaW50ZXJvdmVyJyxcbiAgICAgICAgJ21zcG9pbnRlcnVwJyxcbiAgICAgICAgJ3BvaW50ZXJvdXQnLFxuICAgICAgICAnbXNzaXRlbW9kZWp1bXBsaXN0aXRlbXJlbW92ZWQnLFxuICAgICAgICAnbXN0aHVtYm5haWxjbGljaycsXG4gICAgICAgICdzdG9wJyxcbiAgICAgICAgJ3N0b3JhZ2Vjb21taXQnXG4gICAgXTtcbiAgICB2YXIgd2ViZ2xFdmVudE5hbWVzID0gWyd3ZWJnbGNvbnRleHRyZXN0b3JlZCcsICd3ZWJnbGNvbnRleHRsb3N0JywgJ3dlYmdsY29udGV4dGNyZWF0aW9uZXJyb3InXTtcbiAgICB2YXIgZm9ybUV2ZW50TmFtZXMgPSBbJ2F1dG9jb21wbGV0ZScsICdhdXRvY29tcGxldGVlcnJvciddO1xuICAgIHZhciBkZXRhaWxFdmVudE5hbWVzID0gWyd0b2dnbGUnXTtcbiAgICB2YXIgZXZlbnROYW1lcyA9IF9fc3ByZWFkQXJyYXkoX19zcHJlYWRBcnJheShfX3NwcmVhZEFycmF5KF9fc3ByZWFkQXJyYXkoX19zcHJlYWRBcnJheShfX3NwcmVhZEFycmF5KF9fc3ByZWFkQXJyYXkoX19zcHJlYWRBcnJheShbXSwgZ2xvYmFsRXZlbnRIYW5kbGVyc0V2ZW50TmFtZXMsIHRydWUpLCB3ZWJnbEV2ZW50TmFtZXMsIHRydWUpLCBmb3JtRXZlbnROYW1lcywgdHJ1ZSksIGRldGFpbEV2ZW50TmFtZXMsIHRydWUpLCBkb2N1bWVudEV2ZW50TmFtZXMsIHRydWUpLCB3aW5kb3dFdmVudE5hbWVzLCB0cnVlKSwgaHRtbEVsZW1lbnRFdmVudE5hbWVzLCB0cnVlKSwgaWVFbGVtZW50RXZlbnROYW1lcywgdHJ1ZSk7XG4gICAgLy8gV2hlbmV2ZXIgYW55IGV2ZW50TGlzdGVuZXIgZmlyZXMsIHdlIGNoZWNrIHRoZSBldmVudExpc3RlbmVyIHRhcmdldCBhbmQgYWxsIHBhcmVudHNcbiAgICAvLyBmb3IgYG9ud2hhdGV2ZXJgIHByb3BlcnRpZXMgYW5kIHJlcGxhY2UgdGhlbSB3aXRoIHpvbmUtYm91bmQgZnVuY3Rpb25zXG4gICAgLy8gLSBDaHJvbWUgKGZvciBub3cpXG4gICAgZnVuY3Rpb24gcGF0Y2hWaWFDYXB0dXJpbmdBbGxUaGVFdmVudHMoYXBpKSB7XG4gICAgICAgIHZhciB1bmJvdW5kS2V5ID0gYXBpLnN5bWJvbCgndW5ib3VuZCcpO1xuICAgICAgICB2YXIgX2xvb3BfNSA9IGZ1bmN0aW9uIChpKSB7XG4gICAgICAgICAgICB2YXIgcHJvcGVydHkgPSBldmVudE5hbWVzW2ldO1xuICAgICAgICAgICAgdmFyIG9ucHJvcGVydHkgPSAnb24nICsgcHJvcGVydHk7XG4gICAgICAgICAgICBzZWxmLmFkZEV2ZW50TGlzdGVuZXIocHJvcGVydHksIGZ1bmN0aW9uIChldmVudCkge1xuICAgICAgICAgICAgICAgIHZhciBlbHQgPSBldmVudC50YXJnZXQsIGJvdW5kLCBzb3VyY2U7XG4gICAgICAgICAgICAgICAgaWYgKGVsdCkge1xuICAgICAgICAgICAgICAgICAgICBzb3VyY2UgPSBlbHQuY29uc3RydWN0b3JbJ25hbWUnXSArICcuJyArIG9ucHJvcGVydHk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBzb3VyY2UgPSAndW5rbm93bi4nICsgb25wcm9wZXJ0eTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgd2hpbGUgKGVsdCkge1xuICAgICAgICAgICAgICAgICAgICBpZiAoZWx0W29ucHJvcGVydHldICYmICFlbHRbb25wcm9wZXJ0eV1bdW5ib3VuZEtleV0pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGJvdW5kID0gYXBpLndyYXBXaXRoQ3VycmVudFpvbmUoZWx0W29ucHJvcGVydHldLCBzb3VyY2UpO1xuICAgICAgICAgICAgICAgICAgICAgICAgYm91bmRbdW5ib3VuZEtleV0gPSBlbHRbb25wcm9wZXJ0eV07XG4gICAgICAgICAgICAgICAgICAgICAgICBlbHRbb25wcm9wZXJ0eV0gPSBib3VuZDtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHQgPSBlbHQucGFyZW50RWxlbWVudDtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LCB0cnVlKTtcbiAgICAgICAgfTtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBldmVudE5hbWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICBfbG9vcF81KGkpO1xuICAgICAgICB9XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJlZ2lzdGVyRWxlbWVudFBhdGNoKF9nbG9iYWwsIGFwaSkge1xuICAgICAgICB2YXIgX2IgPSBhcGkuZ2V0R2xvYmFsT2JqZWN0cygpLCBpc0Jyb3dzZXIgPSBfYi5pc0Jyb3dzZXIsIGlzTWl4ID0gX2IuaXNNaXg7XG4gICAgICAgIGlmICgoIWlzQnJvd3NlciAmJiAhaXNNaXgpIHx8ICEoJ3JlZ2lzdGVyRWxlbWVudCcgaW4gX2dsb2JhbC5kb2N1bWVudCkpIHtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICB2YXIgY2FsbGJhY2tzID0gWydjcmVhdGVkQ2FsbGJhY2snLCAnYXR0YWNoZWRDYWxsYmFjaycsICdkZXRhY2hlZENhbGxiYWNrJywgJ2F0dHJpYnV0ZUNoYW5nZWRDYWxsYmFjayddO1xuICAgICAgICBhcGkucGF0Y2hDYWxsYmFja3MoYXBpLCBkb2N1bWVudCwgJ0RvY3VtZW50JywgJ3JlZ2lzdGVyRWxlbWVudCcsIGNhbGxiYWNrcyk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIChmdW5jdGlvbiAoX2dsb2JhbCkge1xuICAgICAgICB2YXIgc3ltYm9sUHJlZml4ID0gX2dsb2JhbFsnX19ab25lX3N5bWJvbF9wcmVmaXgnXSB8fCAnX196b25lX3N5bWJvbF9fJztcbiAgICAgICAgZnVuY3Rpb24gX19zeW1ib2xfXyhuYW1lKSB7XG4gICAgICAgICAgICByZXR1cm4gc3ltYm9sUHJlZml4ICsgbmFtZTtcbiAgICAgICAgfVxuICAgICAgICBfZ2xvYmFsW19fc3ltYm9sX18oJ2xlZ2FjeVBhdGNoJyldID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgdmFyIFpvbmUgPSBfZ2xvYmFsWydab25lJ107XG4gICAgICAgICAgICBab25lLl9fbG9hZF9wYXRjaCgnZGVmaW5lUHJvcGVydHknLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lLCBhcGkpIHtcbiAgICAgICAgICAgICAgICBhcGkuX3JlZGVmaW5lUHJvcGVydHkgPSBfcmVkZWZpbmVQcm9wZXJ0eTtcbiAgICAgICAgICAgICAgICBwcm9wZXJ0eVBhdGNoKCk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIFpvbmUuX19sb2FkX3BhdGNoKCdyZWdpc3RlckVsZW1lbnQnLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lLCBhcGkpIHtcbiAgICAgICAgICAgICAgICByZWdpc3RlckVsZW1lbnRQYXRjaChnbG9iYWwsIGFwaSk7XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIFpvbmUuX19sb2FkX3BhdGNoKCdFdmVudFRhcmdldExlZ2FjeScsIGZ1bmN0aW9uIChnbG9iYWwsIFpvbmUsIGFwaSkge1xuICAgICAgICAgICAgICAgIGV2ZW50VGFyZ2V0TGVnYWN5UGF0Y2goZ2xvYmFsLCBhcGkpO1xuICAgICAgICAgICAgICAgIHByb3BlcnR5RGVzY3JpcHRvckxlZ2FjeVBhdGNoKGFwaSwgZ2xvYmFsKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9O1xuICAgIH0pKHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnID9cbiAgICAgICAgd2luZG93IDpcbiAgICAgICAgdHlwZW9mIGdsb2JhbCAhPT0gJ3VuZGVmaW5lZCcgPyBnbG9iYWwgOiB0eXBlb2Ygc2VsZiAhPT0gJ3VuZGVmaW5lZCcgPyBzZWxmIDoge30pO1xuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIHZhciB0YXNrU3ltYm9sID0gem9uZVN5bWJvbCQxKCd6b25lVGFzaycpO1xuICAgIGZ1bmN0aW9uIHBhdGNoVGltZXIod2luZG93LCBzZXROYW1lLCBjYW5jZWxOYW1lLCBuYW1lU3VmZml4KSB7XG4gICAgICAgIHZhciBzZXROYXRpdmUgPSBudWxsO1xuICAgICAgICB2YXIgY2xlYXJOYXRpdmUgPSBudWxsO1xuICAgICAgICBzZXROYW1lICs9IG5hbWVTdWZmaXg7XG4gICAgICAgIGNhbmNlbE5hbWUgKz0gbmFtZVN1ZmZpeDtcbiAgICAgICAgdmFyIHRhc2tzQnlIYW5kbGVJZCA9IHt9O1xuICAgICAgICBmdW5jdGlvbiBzY2hlZHVsZVRhc2sodGFzaykge1xuICAgICAgICAgICAgdmFyIGRhdGEgPSB0YXNrLmRhdGE7XG4gICAgICAgICAgICBkYXRhLmFyZ3NbMF0gPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRhc2suaW52b2tlLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgZGF0YS5oYW5kbGVJZCA9IHNldE5hdGl2ZS5hcHBseSh3aW5kb3csIGRhdGEuYXJncyk7XG4gICAgICAgICAgICByZXR1cm4gdGFzaztcbiAgICAgICAgfVxuICAgICAgICBmdW5jdGlvbiBjbGVhclRhc2sodGFzaykge1xuICAgICAgICAgICAgcmV0dXJuIGNsZWFyTmF0aXZlLmNhbGwod2luZG93LCB0YXNrLmRhdGEuaGFuZGxlSWQpO1xuICAgICAgICB9XG4gICAgICAgIHNldE5hdGl2ZSA9XG4gICAgICAgICAgICBwYXRjaE1ldGhvZCh3aW5kb3csIHNldE5hbWUsIGZ1bmN0aW9uIChkZWxlZ2F0ZSkgeyByZXR1cm4gZnVuY3Rpb24gKHNlbGYsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGFyZ3NbMF0gPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICAgICAgdmFyIG9wdGlvbnNfMSA9IHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlzUGVyaW9kaWM6IG5hbWVTdWZmaXggPT09ICdJbnRlcnZhbCcsXG4gICAgICAgICAgICAgICAgICAgICAgICBkZWxheTogKG5hbWVTdWZmaXggPT09ICdUaW1lb3V0JyB8fCBuYW1lU3VmZml4ID09PSAnSW50ZXJ2YWwnKSA/IGFyZ3NbMV0gfHwgMCA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdW5kZWZpbmVkLFxuICAgICAgICAgICAgICAgICAgICAgICAgYXJnczogYXJnc1xuICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICB2YXIgY2FsbGJhY2tfMSA9IGFyZ3NbMF07XG4gICAgICAgICAgICAgICAgICAgIGFyZ3NbMF0gPSBmdW5jdGlvbiB0aW1lcigpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGNhbGxiYWNrXzEuYXBwbHkodGhpcywgYXJndW1lbnRzKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGZpbmFsbHkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGlzc3VlLTkzNCwgdGFzayB3aWxsIGJlIGNhbmNlbGxlZFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGV2ZW4gaXQgaXMgYSBwZXJpb2RpYyB0YXNrIHN1Y2ggYXNcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBzZXRJbnRlcnZhbFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hbmd1bGFyL2FuZ3VsYXIvaXNzdWVzLzQwMzg3XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gQ2xlYW51cCB0YXNrc0J5SGFuZGxlSWQgc2hvdWxkIGJlIGhhbmRsZWQgYmVmb3JlIHNjaGVkdWxlVGFza1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIFNpbmNlIHNvbWUgem9uZVNwZWMgbWF5IGludGVyY2VwdCBhbmQgZG9lc24ndCB0cmlnZ2VyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gc2NoZWR1bGVGbihzY2hlZHVsZVRhc2spIHByb3ZpZGVkIGhlcmUuXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCEob3B0aW9uc18xLmlzUGVyaW9kaWMpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2Ygb3B0aW9uc18xLmhhbmRsZUlkID09PSAnbnVtYmVyJykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gaW4gbm9uLW5vZGVqcyBlbnYsIHdlIHJlbW92ZSB0aW1lcklkXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBmcm9tIGxvY2FsIGNhY2hlXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkZWxldGUgdGFza3NCeUhhbmRsZUlkW29wdGlvbnNfMS5oYW5kbGVJZF07XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9uc18xLmhhbmRsZUlkKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBOb2RlIHJldHVybnMgY29tcGxleCBvYmplY3RzIGFzIGhhbmRsZUlkc1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gd2UgcmVtb3ZlIHRhc2sgcmVmZXJlbmNlIGZyb20gdGltZXIgb2JqZWN0XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvcHRpb25zXzEuaGFuZGxlSWRbdGFza1N5bWJvbF0gPSBudWxsO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICB2YXIgdGFzayA9IHNjaGVkdWxlTWFjcm9UYXNrV2l0aEN1cnJlbnRab25lKHNldE5hbWUsIGFyZ3NbMF0sIG9wdGlvbnNfMSwgc2NoZWR1bGVUYXNrLCBjbGVhclRhc2spO1xuICAgICAgICAgICAgICAgICAgICBpZiAoIXRhc2spIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiB0YXNrO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIC8vIE5vZGUuanMgbXVzdCBhZGRpdGlvbmFsbHkgc3VwcG9ydCB0aGUgcmVmIGFuZCB1bnJlZiBmdW5jdGlvbnMuXG4gICAgICAgICAgICAgICAgICAgIHZhciBoYW5kbGUgPSB0YXNrLmRhdGEuaGFuZGxlSWQ7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgaGFuZGxlID09PSAnbnVtYmVyJykge1xuICAgICAgICAgICAgICAgICAgICAgICAgLy8gZm9yIG5vbiBub2RlanMgZW52LCB3ZSBzYXZlIGhhbmRsZUlkOiB0YXNrXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBtYXBwaW5nIGluIGxvY2FsIGNhY2hlIGZvciBjbGVhclRpbWVvdXRcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhc2tzQnlIYW5kbGVJZFtoYW5kbGVdID0gdGFzaztcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChoYW5kbGUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIGZvciBub2RlanMgZW52LCB3ZSBzYXZlIHRhc2tcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHJlZmVyZW5jZSBpbiB0aW1lcklkIE9iamVjdCBmb3IgY2xlYXJUaW1lb3V0XG4gICAgICAgICAgICAgICAgICAgICAgICBoYW5kbGVbdGFza1N5bWJvbF0gPSB0YXNrO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIC8vIGNoZWNrIHdoZXRoZXIgaGFuZGxlIGlzIG51bGwsIGJlY2F1c2Ugc29tZSBwb2x5ZmlsbCBvciBicm93c2VyXG4gICAgICAgICAgICAgICAgICAgIC8vIG1heSByZXR1cm4gdW5kZWZpbmVkIGZyb20gc2V0VGltZW91dC9zZXRJbnRlcnZhbC9zZXRJbW1lZGlhdGUvcmVxdWVzdEFuaW1hdGlvbkZyYW1lXG4gICAgICAgICAgICAgICAgICAgIGlmIChoYW5kbGUgJiYgaGFuZGxlLnJlZiAmJiBoYW5kbGUudW5yZWYgJiYgdHlwZW9mIGhhbmRsZS5yZWYgPT09ICdmdW5jdGlvbicgJiZcbiAgICAgICAgICAgICAgICAgICAgICAgIHR5cGVvZiBoYW5kbGUudW5yZWYgPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhc2sucmVmID0gaGFuZGxlLnJlZi5iaW5kKGhhbmRsZSk7XG4gICAgICAgICAgICAgICAgICAgICAgICB0YXNrLnVucmVmID0gaGFuZGxlLnVucmVmLmJpbmQoaGFuZGxlKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGhhbmRsZSA9PT0gJ251bWJlcicgfHwgaGFuZGxlKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gaGFuZGxlO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0YXNrO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gY2F1c2UgYW4gZXJyb3IgYnkgY2FsbGluZyBpdCBkaXJlY3RseS5cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGRlbGVnYXRlLmFwcGx5KHdpbmRvdywgYXJncyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfTsgfSk7XG4gICAgICAgIGNsZWFyTmF0aXZlID1cbiAgICAgICAgICAgIHBhdGNoTWV0aG9kKHdpbmRvdywgY2FuY2VsTmFtZSwgZnVuY3Rpb24gKGRlbGVnYXRlKSB7IHJldHVybiBmdW5jdGlvbiAoc2VsZiwgYXJncykge1xuICAgICAgICAgICAgICAgIHZhciBpZCA9IGFyZ3NbMF07XG4gICAgICAgICAgICAgICAgdmFyIHRhc2s7XG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZiBpZCA9PT0gJ251bWJlcicpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gbm9uIG5vZGVqcyBlbnYuXG4gICAgICAgICAgICAgICAgICAgIHRhc2sgPSB0YXNrc0J5SGFuZGxlSWRbaWRdO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gbm9kZWpzIGVudi5cbiAgICAgICAgICAgICAgICAgICAgdGFzayA9IGlkICYmIGlkW3Rhc2tTeW1ib2xdO1xuICAgICAgICAgICAgICAgICAgICAvLyBvdGhlciBlbnZpcm9ubWVudHMuXG4gICAgICAgICAgICAgICAgICAgIGlmICghdGFzaykge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGFzayA9IGlkO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmICh0YXNrICYmIHR5cGVvZiB0YXNrLnR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0YXNrLnN0YXRlICE9PSAnbm90U2NoZWR1bGVkJyAmJlxuICAgICAgICAgICAgICAgICAgICAgICAgKHRhc2suY2FuY2VsRm4gJiYgdGFzay5kYXRhLmlzUGVyaW9kaWMgfHwgdGFzay5ydW5Db3VudCA9PT0gMCkpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgaWQgPT09ICdudW1iZXInKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZGVsZXRlIHRhc2tzQnlIYW5kbGVJZFtpZF07XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChpZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlkW3Rhc2tTeW1ib2xdID0gbnVsbDtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIERvIG5vdCBjYW5jZWwgYWxyZWFkeSBjYW5jZWxlZCBmdW5jdGlvbnNcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suem9uZS5jYW5jZWxUYXNrKHRhc2spO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAvLyBjYXVzZSBhbiBlcnJvciBieSBjYWxsaW5nIGl0IGRpcmVjdGx5LlxuICAgICAgICAgICAgICAgICAgICBkZWxlZ2F0ZS5hcHBseSh3aW5kb3csIGFyZ3MpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH07IH0pO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBAbGljZW5zZVxuICAgICAqIENvcHlyaWdodCBHb29nbGUgTExDIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gICAgICpcbiAgICAgKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICAgICAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBwYXRjaEN1c3RvbUVsZW1lbnRzKF9nbG9iYWwsIGFwaSkge1xuICAgICAgICB2YXIgX2IgPSBhcGkuZ2V0R2xvYmFsT2JqZWN0cygpLCBpc0Jyb3dzZXIgPSBfYi5pc0Jyb3dzZXIsIGlzTWl4ID0gX2IuaXNNaXg7XG4gICAgICAgIGlmICgoIWlzQnJvd3NlciAmJiAhaXNNaXgpIHx8ICFfZ2xvYmFsWydjdXN0b21FbGVtZW50cyddIHx8ICEoJ2N1c3RvbUVsZW1lbnRzJyBpbiBfZ2xvYmFsKSkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHZhciBjYWxsYmFja3MgPSBbJ2Nvbm5lY3RlZENhbGxiYWNrJywgJ2Rpc2Nvbm5lY3RlZENhbGxiYWNrJywgJ2Fkb3B0ZWRDYWxsYmFjaycsICdhdHRyaWJ1dGVDaGFuZ2VkQ2FsbGJhY2snXTtcbiAgICAgICAgYXBpLnBhdGNoQ2FsbGJhY2tzKGFwaSwgX2dsb2JhbC5jdXN0b21FbGVtZW50cywgJ2N1c3RvbUVsZW1lbnRzJywgJ2RlZmluZScsIGNhbGxiYWNrcyk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEBsaWNlbnNlXG4gICAgICogQ29weXJpZ2h0IEdvb2dsZSBMTEMgQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAgICAgKlxuICAgICAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gICAgICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGV2ZW50VGFyZ2V0UGF0Y2goX2dsb2JhbCwgYXBpKSB7XG4gICAgICAgIGlmIChab25lW2FwaS5zeW1ib2woJ3BhdGNoRXZlbnRUYXJnZXQnKV0pIHtcbiAgICAgICAgICAgIC8vIEV2ZW50VGFyZ2V0IGlzIGFscmVhZHkgcGF0Y2hlZC5cbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICB2YXIgX2IgPSBhcGkuZ2V0R2xvYmFsT2JqZWN0cygpLCBldmVudE5hbWVzID0gX2IuZXZlbnROYW1lcywgem9uZVN5bWJvbEV2ZW50TmFtZXMgPSBfYi56b25lU3ltYm9sRXZlbnROYW1lcywgVFJVRV9TVFIgPSBfYi5UUlVFX1NUUiwgRkFMU0VfU1RSID0gX2IuRkFMU0VfU1RSLCBaT05FX1NZTUJPTF9QUkVGSVggPSBfYi5aT05FX1NZTUJPTF9QUkVGSVg7XG4gICAgICAgIC8vICBwcmVkZWZpbmUgYWxsIF9fem9uZV9zeW1ib2xfXyArIGV2ZW50TmFtZSArIHRydWUvZmFsc2Ugc3RyaW5nXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgZXZlbnROYW1lcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgdmFyIGV2ZW50TmFtZSA9IGV2ZW50TmFtZXNbaV07XG4gICAgICAgICAgICB2YXIgZmFsc2VFdmVudE5hbWUgPSBldmVudE5hbWUgKyBGQUxTRV9TVFI7XG4gICAgICAgICAgICB2YXIgdHJ1ZUV2ZW50TmFtZSA9IGV2ZW50TmFtZSArIFRSVUVfU1RSO1xuICAgICAgICAgICAgdmFyIHN5bWJvbCA9IFpPTkVfU1lNQk9MX1BSRUZJWCArIGZhbHNlRXZlbnROYW1lO1xuICAgICAgICAgICAgdmFyIHN5bWJvbENhcHR1cmUgPSBaT05FX1NZTUJPTF9QUkVGSVggKyB0cnVlRXZlbnROYW1lO1xuICAgICAgICAgICAgem9uZVN5bWJvbEV2ZW50TmFtZXNbZXZlbnROYW1lXSA9IHt9O1xuICAgICAgICAgICAgem9uZVN5bWJvbEV2ZW50TmFtZXNbZXZlbnROYW1lXVtGQUxTRV9TVFJdID0gc3ltYm9sO1xuICAgICAgICAgICAgem9uZVN5bWJvbEV2ZW50TmFtZXNbZXZlbnROYW1lXVtUUlVFX1NUUl0gPSBzeW1ib2xDYXB0dXJlO1xuICAgICAgICB9XG4gICAgICAgIHZhciBFVkVOVF9UQVJHRVQgPSBfZ2xvYmFsWydFdmVudFRhcmdldCddO1xuICAgICAgICBpZiAoIUVWRU5UX1RBUkdFVCB8fCAhRVZFTlRfVEFSR0VULnByb3RvdHlwZSkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGFwaS5wYXRjaEV2ZW50VGFyZ2V0KF9nbG9iYWwsIGFwaSwgW0VWRU5UX1RBUkdFVCAmJiBFVkVOVF9UQVJHRVQucHJvdG90eXBlXSk7XG4gICAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICBmdW5jdGlvbiBwYXRjaEV2ZW50KGdsb2JhbCwgYXBpKSB7XG4gICAgICAgIGFwaS5wYXRjaEV2ZW50UHJvdG90eXBlKGdsb2JhbCwgYXBpKTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQGxpY2Vuc2VcbiAgICAgKiBDb3B5cmlnaHQgR29vZ2xlIExMQyBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICAgICAqXG4gICAgICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAgICAgKiBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIGF0IGh0dHBzOi8vYW5ndWxhci5pby9saWNlbnNlXG4gICAgICovXG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ2xlZ2FjeScsIGZ1bmN0aW9uIChnbG9iYWwpIHtcbiAgICAgICAgdmFyIGxlZ2FjeVBhdGNoID0gZ2xvYmFsW1pvbmUuX19zeW1ib2xfXygnbGVnYWN5UGF0Y2gnKV07XG4gICAgICAgIGlmIChsZWdhY3lQYXRjaCkge1xuICAgICAgICAgICAgbGVnYWN5UGF0Y2goKTtcbiAgICAgICAgfVxuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdxdWV1ZU1pY3JvdGFzaycsIGZ1bmN0aW9uIChnbG9iYWwsIFpvbmUsIGFwaSkge1xuICAgICAgICBhcGkucGF0Y2hNZXRob2QoZ2xvYmFsLCAncXVldWVNaWNyb3Rhc2snLCBmdW5jdGlvbiAoZGVsZWdhdGUpIHtcbiAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbiAoc2VsZiwgYXJncykge1xuICAgICAgICAgICAgICAgIFpvbmUuY3VycmVudC5zY2hlZHVsZU1pY3JvVGFzaygncXVldWVNaWNyb3Rhc2snLCBhcmdzWzBdKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgIH0pO1xuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCd0aW1lcnMnLCBmdW5jdGlvbiAoZ2xvYmFsKSB7XG4gICAgICAgIHZhciBzZXQgPSAnc2V0JztcbiAgICAgICAgdmFyIGNsZWFyID0gJ2NsZWFyJztcbiAgICAgICAgcGF0Y2hUaW1lcihnbG9iYWwsIHNldCwgY2xlYXIsICdUaW1lb3V0Jyk7XG4gICAgICAgIHBhdGNoVGltZXIoZ2xvYmFsLCBzZXQsIGNsZWFyLCAnSW50ZXJ2YWwnKTtcbiAgICAgICAgcGF0Y2hUaW1lcihnbG9iYWwsIHNldCwgY2xlYXIsICdJbW1lZGlhdGUnKTtcbiAgICB9KTtcbiAgICBab25lLl9fbG9hZF9wYXRjaCgncmVxdWVzdEFuaW1hdGlvbkZyYW1lJywgZnVuY3Rpb24gKGdsb2JhbCkge1xuICAgICAgICBwYXRjaFRpbWVyKGdsb2JhbCwgJ3JlcXVlc3QnLCAnY2FuY2VsJywgJ0FuaW1hdGlvbkZyYW1lJyk7XG4gICAgICAgIHBhdGNoVGltZXIoZ2xvYmFsLCAnbW96UmVxdWVzdCcsICdtb3pDYW5jZWwnLCAnQW5pbWF0aW9uRnJhbWUnKTtcbiAgICAgICAgcGF0Y2hUaW1lcihnbG9iYWwsICd3ZWJraXRSZXF1ZXN0JywgJ3dlYmtpdENhbmNlbCcsICdBbmltYXRpb25GcmFtZScpO1xuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdibG9ja2luZycsIGZ1bmN0aW9uIChnbG9iYWwsIFpvbmUpIHtcbiAgICAgICAgdmFyIGJsb2NraW5nTWV0aG9kcyA9IFsnYWxlcnQnLCAncHJvbXB0JywgJ2NvbmZpcm0nXTtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBibG9ja2luZ01ldGhvZHMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgIHZhciBuYW1lXzIgPSBibG9ja2luZ01ldGhvZHNbaV07XG4gICAgICAgICAgICBwYXRjaE1ldGhvZChnbG9iYWwsIG5hbWVfMiwgZnVuY3Rpb24gKGRlbGVnYXRlLCBzeW1ib2wsIG5hbWUpIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gZnVuY3Rpb24gKHMsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFpvbmUuY3VycmVudC5ydW4oZGVsZWdhdGUsIGdsb2JhbCwgYXJncywgbmFtZSk7XG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfSk7XG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ0V2ZW50VGFyZ2V0JywgZnVuY3Rpb24gKGdsb2JhbCwgWm9uZSwgYXBpKSB7XG4gICAgICAgIHBhdGNoRXZlbnQoZ2xvYmFsLCBhcGkpO1xuICAgICAgICBldmVudFRhcmdldFBhdGNoKGdsb2JhbCwgYXBpKTtcbiAgICAgICAgLy8gcGF0Y2ggWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldCdzIGFkZEV2ZW50TGlzdGVuZXIvcmVtb3ZlRXZlbnRMaXN0ZW5lclxuICAgICAgICB2YXIgWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldCA9IGdsb2JhbFsnWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldCddO1xuICAgICAgICBpZiAoWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldCAmJiBYTUxIdHRwUmVxdWVzdEV2ZW50VGFyZ2V0LnByb3RvdHlwZSkge1xuICAgICAgICAgICAgYXBpLnBhdGNoRXZlbnRUYXJnZXQoZ2xvYmFsLCBhcGksIFtYTUxIdHRwUmVxdWVzdEV2ZW50VGFyZ2V0LnByb3RvdHlwZV0pO1xuICAgICAgICB9XG4gICAgfSk7XG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ011dGF0aW9uT2JzZXJ2ZXInLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lLCBhcGkpIHtcbiAgICAgICAgcGF0Y2hDbGFzcygnTXV0YXRpb25PYnNlcnZlcicpO1xuICAgICAgICBwYXRjaENsYXNzKCdXZWJLaXRNdXRhdGlvbk9ic2VydmVyJyk7XG4gICAgfSk7XG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ0ludGVyc2VjdGlvbk9ic2VydmVyJywgZnVuY3Rpb24gKGdsb2JhbCwgWm9uZSwgYXBpKSB7XG4gICAgICAgIHBhdGNoQ2xhc3MoJ0ludGVyc2VjdGlvbk9ic2VydmVyJyk7XG4gICAgfSk7XG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ0ZpbGVSZWFkZXInLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lLCBhcGkpIHtcbiAgICAgICAgcGF0Y2hDbGFzcygnRmlsZVJlYWRlcicpO1xuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdvbl9wcm9wZXJ0eScsIGZ1bmN0aW9uIChnbG9iYWwsIFpvbmUsIGFwaSkge1xuICAgICAgICBwcm9wZXJ0eURlc2NyaXB0b3JQYXRjaChhcGksIGdsb2JhbCk7XG4gICAgfSk7XG4gICAgWm9uZS5fX2xvYWRfcGF0Y2goJ2N1c3RvbUVsZW1lbnRzJywgZnVuY3Rpb24gKGdsb2JhbCwgWm9uZSwgYXBpKSB7XG4gICAgICAgIHBhdGNoQ3VzdG9tRWxlbWVudHMoZ2xvYmFsLCBhcGkpO1xuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdYSFInLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lKSB7XG4gICAgICAgIC8vIFRyZWF0IFhNTEh0dHBSZXF1ZXN0IGFzIGEgbWFjcm90YXNrLlxuICAgICAgICBwYXRjaFhIUihnbG9iYWwpO1xuICAgICAgICB2YXIgWEhSX1RBU0sgPSB6b25lU3ltYm9sJDEoJ3hoclRhc2snKTtcbiAgICAgICAgdmFyIFhIUl9TWU5DID0gem9uZVN5bWJvbCQxKCd4aHJTeW5jJyk7XG4gICAgICAgIHZhciBYSFJfTElTVEVORVIgPSB6b25lU3ltYm9sJDEoJ3hockxpc3RlbmVyJyk7XG4gICAgICAgIHZhciBYSFJfU0NIRURVTEVEID0gem9uZVN5bWJvbCQxKCd4aHJTY2hlZHVsZWQnKTtcbiAgICAgICAgdmFyIFhIUl9VUkwgPSB6b25lU3ltYm9sJDEoJ3hoclVSTCcpO1xuICAgICAgICB2YXIgWEhSX0VSUk9SX0JFRk9SRV9TQ0hFRFVMRUQgPSB6b25lU3ltYm9sJDEoJ3hockVycm9yQmVmb3JlU2NoZWR1bGVkJyk7XG4gICAgICAgIGZ1bmN0aW9uIHBhdGNoWEhSKHdpbmRvdykge1xuICAgICAgICAgICAgdmFyIFhNTEh0dHBSZXF1ZXN0ID0gd2luZG93WydYTUxIdHRwUmVxdWVzdCddO1xuICAgICAgICAgICAgaWYgKCFYTUxIdHRwUmVxdWVzdCkge1xuICAgICAgICAgICAgICAgIC8vIFhNTEh0dHBSZXF1ZXN0IGlzIG5vdCBhdmFpbGFibGUgaW4gc2VydmljZSB3b3JrZXJcbiAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgWE1MSHR0cFJlcXVlc3RQcm90b3R5cGUgPSBYTUxIdHRwUmVxdWVzdC5wcm90b3R5cGU7XG4gICAgICAgICAgICBmdW5jdGlvbiBmaW5kUGVuZGluZ1Rhc2sodGFyZ2V0KSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRhcmdldFtYSFJfVEFTS107XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgb3JpQWRkTGlzdGVuZXIgPSBYTUxIdHRwUmVxdWVzdFByb3RvdHlwZVtaT05FX1NZTUJPTF9BRERfRVZFTlRfTElTVEVORVJdO1xuICAgICAgICAgICAgdmFyIG9yaVJlbW92ZUxpc3RlbmVyID0gWE1MSHR0cFJlcXVlc3RQcm90b3R5cGVbWk9ORV9TWU1CT0xfUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXTtcbiAgICAgICAgICAgIGlmICghb3JpQWRkTGlzdGVuZXIpIHtcbiAgICAgICAgICAgICAgICB2YXIgWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldF8xID0gd2luZG93WydYTUxIdHRwUmVxdWVzdEV2ZW50VGFyZ2V0J107XG4gICAgICAgICAgICAgICAgaWYgKFhNTEh0dHBSZXF1ZXN0RXZlbnRUYXJnZXRfMSkge1xuICAgICAgICAgICAgICAgICAgICB2YXIgWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldFByb3RvdHlwZSA9IFhNTEh0dHBSZXF1ZXN0RXZlbnRUYXJnZXRfMS5wcm90b3R5cGU7XG4gICAgICAgICAgICAgICAgICAgIG9yaUFkZExpc3RlbmVyID0gWE1MSHR0cFJlcXVlc3RFdmVudFRhcmdldFByb3RvdHlwZVtaT05FX1NZTUJPTF9BRERfRVZFTlRfTElTVEVORVJdO1xuICAgICAgICAgICAgICAgICAgICBvcmlSZW1vdmVMaXN0ZW5lciA9IFhNTEh0dHBSZXF1ZXN0RXZlbnRUYXJnZXRQcm90b3R5cGVbWk9ORV9TWU1CT0xfUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgUkVBRFlfU1RBVEVfQ0hBTkdFID0gJ3JlYWR5c3RhdGVjaGFuZ2UnO1xuICAgICAgICAgICAgdmFyIFNDSEVEVUxFRCA9ICdzY2hlZHVsZWQnO1xuICAgICAgICAgICAgZnVuY3Rpb24gc2NoZWR1bGVUYXNrKHRhc2spIHtcbiAgICAgICAgICAgICAgICB2YXIgZGF0YSA9IHRhc2suZGF0YTtcbiAgICAgICAgICAgICAgICB2YXIgdGFyZ2V0ID0gZGF0YS50YXJnZXQ7XG4gICAgICAgICAgICAgICAgdGFyZ2V0W1hIUl9TQ0hFRFVMRURdID0gZmFsc2U7XG4gICAgICAgICAgICAgICAgdGFyZ2V0W1hIUl9FUlJPUl9CRUZPUkVfU0NIRURVTEVEXSA9IGZhbHNlO1xuICAgICAgICAgICAgICAgIC8vIHJlbW92ZSBleGlzdGluZyBldmVudCBsaXN0ZW5lclxuICAgICAgICAgICAgICAgIHZhciBsaXN0ZW5lciA9IHRhcmdldFtYSFJfTElTVEVORVJdO1xuICAgICAgICAgICAgICAgIGlmICghb3JpQWRkTGlzdGVuZXIpIHtcbiAgICAgICAgICAgICAgICAgICAgb3JpQWRkTGlzdGVuZXIgPSB0YXJnZXRbWk9ORV9TWU1CT0xfQUREX0VWRU5UX0xJU1RFTkVSXTtcbiAgICAgICAgICAgICAgICAgICAgb3JpUmVtb3ZlTGlzdGVuZXIgPSB0YXJnZXRbWk9ORV9TWU1CT0xfUkVNT1ZFX0VWRU5UX0xJU1RFTkVSXTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKGxpc3RlbmVyKSB7XG4gICAgICAgICAgICAgICAgICAgIG9yaVJlbW92ZUxpc3RlbmVyLmNhbGwodGFyZ2V0LCBSRUFEWV9TVEFURV9DSEFOR0UsIGxpc3RlbmVyKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgdmFyIG5ld0xpc3RlbmVyID0gdGFyZ2V0W1hIUl9MSVNURU5FUl0gPSBmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICh0YXJnZXQucmVhZHlTdGF0ZSA9PT0gdGFyZ2V0LkRPTkUpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHNvbWV0aW1lcyBvbiBzb21lIGJyb3dzZXJzIFhNTEh0dHBSZXF1ZXN0IHdpbGwgZmlyZSBvbnJlYWR5c3RhdGVjaGFuZ2Ugd2l0aFxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gcmVhZHlTdGF0ZT00IG11bHRpcGxlIHRpbWVzLCBzbyB3ZSBuZWVkIHRvIGNoZWNrIHRhc2sgc3RhdGUgaGVyZVxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCFkYXRhLmFib3J0ZWQgJiYgdGFyZ2V0W1hIUl9TQ0hFRFVMRURdICYmIHRhc2suc3RhdGUgPT09IFNDSEVEVUxFRCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGNoZWNrIHdoZXRoZXIgdGhlIHhociBoYXMgcmVnaXN0ZXJlZCBvbmxvYWQgbGlzdGVuZXJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBpZiB0aGF0IGlzIHRoZSBjYXNlLCB0aGUgdGFzayBzaG91bGQgaW52b2tlIGFmdGVyIGFsbFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIG9ubG9hZCBsaXN0ZW5lcnMgZmluaXNoLlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIEFsc28gaWYgdGhlIHJlcXVlc3QgZmFpbGVkIHdpdGhvdXQgcmVzcG9uc2UgKHN0YXR1cyA9IDApLCB0aGUgbG9hZCBldmVudCBoYW5kbGVyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gd2lsbCBub3QgYmUgdHJpZ2dlcmVkLCBpbiB0aGF0IGNhc2UsIHdlIHNob3VsZCBhbHNvIGludm9rZSB0aGUgcGxhY2Vob2xkZXIgY2FsbGJhY2tcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyB0byBjbG9zZSB0aGUgWE1MSHR0cFJlcXVlc3Q6OnNlbmQgbWFjcm9UYXNrLlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hbmd1bGFyL2FuZ3VsYXIvaXNzdWVzLzM4Nzk1XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGxvYWRUYXNrcyA9IHRhcmdldFtab25lLl9fc3ltYm9sX18oJ2xvYWRmYWxzZScpXTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodGFyZ2V0LnN0YXR1cyAhPT0gMCAmJiBsb2FkVGFza3MgJiYgbG9hZFRhc2tzLmxlbmd0aCA+IDApIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIG9yaUludm9rZV8xID0gdGFzay5pbnZva2U7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suaW52b2tlID0gZnVuY3Rpb24gKCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gbmVlZCB0byBsb2FkIHRoZSB0YXNrcyBhZ2FpbiwgYmVjYXVzZSBpbiBvdGhlclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gbG9hZCBsaXN0ZW5lciwgdGhleSBtYXkgcmVtb3ZlIHRoZW1zZWx2ZXNcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBsb2FkVGFza3MgPSB0YXJnZXRbWm9uZS5fX3N5bWJvbF9fKCdsb2FkZmFsc2UnKV07XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGxvYWRUYXNrcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChsb2FkVGFza3NbaV0gPT09IHRhc2spIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbG9hZFRhc2tzLnNwbGljZShpLCAxKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoIWRhdGEuYWJvcnRlZCAmJiB0YXNrLnN0YXRlID09PSBTQ0hFRFVMRUQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvcmlJbnZva2VfMS5jYWxsKHRhc2spO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsb2FkVGFza3MucHVzaCh0YXNrKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suaW52b2tlKCk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBpZiAoIWRhdGEuYWJvcnRlZCAmJiB0YXJnZXRbWEhSX1NDSEVEVUxFRF0gPT09IGZhbHNlKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gZXJyb3Igb2NjdXJzIHdoZW4geGhyLnNlbmQoKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhcmdldFtYSFJfRVJST1JfQkVGT1JFX1NDSEVEVUxFRF0gPSB0cnVlO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICBvcmlBZGRMaXN0ZW5lci5jYWxsKHRhcmdldCwgUkVBRFlfU1RBVEVfQ0hBTkdFLCBuZXdMaXN0ZW5lcik7XG4gICAgICAgICAgICAgICAgdmFyIHN0b3JlZFRhc2sgPSB0YXJnZXRbWEhSX1RBU0tdO1xuICAgICAgICAgICAgICAgIGlmICghc3RvcmVkVGFzaykge1xuICAgICAgICAgICAgICAgICAgICB0YXJnZXRbWEhSX1RBU0tdID0gdGFzaztcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgc2VuZE5hdGl2ZS5hcHBseSh0YXJnZXQsIGRhdGEuYXJncyk7XG4gICAgICAgICAgICAgICAgdGFyZ2V0W1hIUl9TQ0hFRFVMRURdID0gdHJ1ZTtcbiAgICAgICAgICAgICAgICByZXR1cm4gdGFzaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZ1bmN0aW9uIHBsYWNlaG9sZGVyQ2FsbGJhY2soKSB7IH1cbiAgICAgICAgICAgIGZ1bmN0aW9uIGNsZWFyVGFzayh0YXNrKSB7XG4gICAgICAgICAgICAgICAgdmFyIGRhdGEgPSB0YXNrLmRhdGE7XG4gICAgICAgICAgICAgICAgLy8gTm90ZSAtIGlkZWFsbHksIHdlIHdvdWxkIGNhbGwgZGF0YS50YXJnZXQucmVtb3ZlRXZlbnRMaXN0ZW5lciBoZXJlLCBidXQgaXQncyB0b28gbGF0ZVxuICAgICAgICAgICAgICAgIC8vIHRvIHByZXZlbnQgaXQgZnJvbSBmaXJpbmcuIFNvIGluc3RlYWQsIHdlIHN0b3JlIGluZm8gZm9yIHRoZSBldmVudCBsaXN0ZW5lci5cbiAgICAgICAgICAgICAgICBkYXRhLmFib3J0ZWQgPSB0cnVlO1xuICAgICAgICAgICAgICAgIHJldHVybiBhYm9ydE5hdGl2ZS5hcHBseShkYXRhLnRhcmdldCwgZGF0YS5hcmdzKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBvcGVuTmF0aXZlID0gcGF0Y2hNZXRob2QoWE1MSHR0cFJlcXVlc3RQcm90b3R5cGUsICdvcGVuJywgZnVuY3Rpb24gKCkgeyByZXR1cm4gZnVuY3Rpb24gKHNlbGYsIGFyZ3MpIHtcbiAgICAgICAgICAgICAgICBzZWxmW1hIUl9TWU5DXSA9IGFyZ3NbMl0gPT0gZmFsc2U7XG4gICAgICAgICAgICAgICAgc2VsZltYSFJfVVJMXSA9IGFyZ3NbMV07XG4gICAgICAgICAgICAgICAgcmV0dXJuIG9wZW5OYXRpdmUuYXBwbHkoc2VsZiwgYXJncyk7XG4gICAgICAgICAgICB9OyB9KTtcbiAgICAgICAgICAgIHZhciBYTUxIVFRQUkVRVUVTVF9TT1VSQ0UgPSAnWE1MSHR0cFJlcXVlc3Quc2VuZCc7XG4gICAgICAgICAgICB2YXIgZmV0Y2hUYXNrQWJvcnRpbmcgPSB6b25lU3ltYm9sJDEoJ2ZldGNoVGFza0Fib3J0aW5nJyk7XG4gICAgICAgICAgICB2YXIgZmV0Y2hUYXNrU2NoZWR1bGluZyA9IHpvbmVTeW1ib2wkMSgnZmV0Y2hUYXNrU2NoZWR1bGluZycpO1xuICAgICAgICAgICAgdmFyIHNlbmROYXRpdmUgPSBwYXRjaE1ldGhvZChYTUxIdHRwUmVxdWVzdFByb3RvdHlwZSwgJ3NlbmQnLCBmdW5jdGlvbiAoKSB7IHJldHVybiBmdW5jdGlvbiAoc2VsZiwgYXJncykge1xuICAgICAgICAgICAgICAgIGlmIChab25lLmN1cnJlbnRbZmV0Y2hUYXNrU2NoZWR1bGluZ10gPT09IHRydWUpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gYSBmZXRjaCBpcyBzY2hlZHVsaW5nLCBzbyB3ZSBhcmUgdXNpbmcgeGhyIHRvIHBvbHlmaWxsIGZldGNoXG4gICAgICAgICAgICAgICAgICAgIC8vIGFuZCBiZWNhdXNlIHdlIGFscmVhZHkgc2NoZWR1bGUgbWFjcm9UYXNrIGZvciBmZXRjaCwgd2Ugc2hvdWxkXG4gICAgICAgICAgICAgICAgICAgIC8vIG5vdCBzY2hlZHVsZSBhIG1hY3JvVGFzayBmb3IgeGhyIGFnYWluXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBzZW5kTmF0aXZlLmFwcGx5KHNlbGYsIGFyZ3MpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBpZiAoc2VsZltYSFJfU1lOQ10pIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gaWYgdGhlIFhIUiBpcyBzeW5jIHRoZXJlIGlzIG5vIHRhc2sgdG8gc2NoZWR1bGUsIGp1c3QgZXhlY3V0ZSB0aGUgY29kZS5cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHNlbmROYXRpdmUuYXBwbHkoc2VsZiwgYXJncyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICB2YXIgb3B0aW9ucyA9IHsgdGFyZ2V0OiBzZWxmLCB1cmw6IHNlbGZbWEhSX1VSTF0sIGlzUGVyaW9kaWM6IGZhbHNlLCBhcmdzOiBhcmdzLCBhYm9ydGVkOiBmYWxzZSB9O1xuICAgICAgICAgICAgICAgICAgICB2YXIgdGFzayA9IHNjaGVkdWxlTWFjcm9UYXNrV2l0aEN1cnJlbnRab25lKFhNTEhUVFBSRVFVRVNUX1NPVVJDRSwgcGxhY2Vob2xkZXJDYWxsYmFjaywgb3B0aW9ucywgc2NoZWR1bGVUYXNrLCBjbGVhclRhc2spO1xuICAgICAgICAgICAgICAgICAgICBpZiAoc2VsZiAmJiBzZWxmW1hIUl9FUlJPUl9CRUZPUkVfU0NIRURVTEVEXSA9PT0gdHJ1ZSAmJiAhb3B0aW9ucy5hYm9ydGVkICYmXG4gICAgICAgICAgICAgICAgICAgICAgICB0YXNrLnN0YXRlID09PSBTQ0hFRFVMRUQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIHhociByZXF1ZXN0IHRocm93IGVycm9yIHdoZW4gc2VuZFxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gd2Ugc2hvdWxkIGludm9rZSB0YXNrIGluc3RlYWQgb2YgbGVhdmluZyBhIHNjaGVkdWxlZFxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gcGVuZGluZyBtYWNyb1Rhc2tcbiAgICAgICAgICAgICAgICAgICAgICAgIHRhc2suaW52b2tlKCk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9OyB9KTtcbiAgICAgICAgICAgIHZhciBhYm9ydE5hdGl2ZSA9IHBhdGNoTWV0aG9kKFhNTEh0dHBSZXF1ZXN0UHJvdG90eXBlLCAnYWJvcnQnLCBmdW5jdGlvbiAoKSB7IHJldHVybiBmdW5jdGlvbiAoc2VsZiwgYXJncykge1xuICAgICAgICAgICAgICAgIHZhciB0YXNrID0gZmluZFBlbmRpbmdUYXNrKHNlbGYpO1xuICAgICAgICAgICAgICAgIGlmICh0YXNrICYmIHR5cGVvZiB0YXNrLnR5cGUgPT0gJ3N0cmluZycpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gSWYgdGhlIFhIUiBoYXMgYWxyZWFkeSBjb21wbGV0ZWQsIGRvIG5vdGhpbmcuXG4gICAgICAgICAgICAgICAgICAgIC8vIElmIHRoZSBYSFIgaGFzIGFscmVhZHkgYmVlbiBhYm9ydGVkLCBkbyBub3RoaW5nLlxuICAgICAgICAgICAgICAgICAgICAvLyBGaXggIzU2OSwgY2FsbCBhYm9ydCBtdWx0aXBsZSB0aW1lcyBiZWZvcmUgZG9uZSB3aWxsIGNhdXNlXG4gICAgICAgICAgICAgICAgICAgIC8vIG1hY3JvVGFzayB0YXNrIGNvdW50IGJlIG5lZ2F0aXZlIG51bWJlclxuICAgICAgICAgICAgICAgICAgICBpZiAodGFzay5jYW5jZWxGbiA9PSBudWxsIHx8ICh0YXNrLmRhdGEgJiYgdGFzay5kYXRhLmFib3J0ZWQpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgdGFzay56b25lLmNhbmNlbFRhc2sodGFzayk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKFpvbmUuY3VycmVudFtmZXRjaFRhc2tBYm9ydGluZ10gPT09IHRydWUpIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gdGhlIGFib3J0IGlzIGNhbGxlZCBmcm9tIGZldGNoIHBvbHlmaWxsLCB3ZSBuZWVkIHRvIGNhbGwgbmF0aXZlIGFib3J0IG9mIFhIUi5cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGFib3J0TmF0aXZlLmFwcGx5KHNlbGYsIGFyZ3MpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAvLyBPdGhlcndpc2UsIHdlIGFyZSB0cnlpbmcgdG8gYWJvcnQgYW4gWEhSIHdoaWNoIGhhcyBub3QgeWV0IGJlZW4gc2VudCwgc28gdGhlcmUgaXMgbm9cbiAgICAgICAgICAgICAgICAvLyB0YXNrXG4gICAgICAgICAgICAgICAgLy8gdG8gY2FuY2VsLiBEbyBub3RoaW5nLlxuICAgICAgICAgICAgfTsgfSk7XG4gICAgICAgIH1cbiAgICB9KTtcbiAgICBab25lLl9fbG9hZF9wYXRjaCgnZ2VvbG9jYXRpb24nLCBmdW5jdGlvbiAoZ2xvYmFsKSB7XG4gICAgICAgIC8vLyBHRU9fTE9DQVRJT05cbiAgICAgICAgaWYgKGdsb2JhbFsnbmF2aWdhdG9yJ10gJiYgZ2xvYmFsWyduYXZpZ2F0b3InXS5nZW9sb2NhdGlvbikge1xuICAgICAgICAgICAgcGF0Y2hQcm90b3R5cGUoZ2xvYmFsWyduYXZpZ2F0b3InXS5nZW9sb2NhdGlvbiwgWydnZXRDdXJyZW50UG9zaXRpb24nLCAnd2F0Y2hQb3NpdGlvbiddKTtcbiAgICAgICAgfVxuICAgIH0pO1xuICAgIFpvbmUuX19sb2FkX3BhdGNoKCdQcm9taXNlUmVqZWN0aW9uRXZlbnQnLCBmdW5jdGlvbiAoZ2xvYmFsLCBab25lKSB7XG4gICAgICAgIC8vIGhhbmRsZSB1bmhhbmRsZWQgcHJvbWlzZSByZWplY3Rpb25cbiAgICAgICAgZnVuY3Rpb24gZmluZFByb21pc2VSZWplY3Rpb25IYW5kbGVyKGV2dE5hbWUpIHtcbiAgICAgICAgICAgIHJldHVybiBmdW5jdGlvbiAoZSkge1xuICAgICAgICAgICAgICAgIHZhciBldmVudFRhc2tzID0gZmluZEV2ZW50VGFza3MoZ2xvYmFsLCBldnROYW1lKTtcbiAgICAgICAgICAgICAgICBldmVudFRhc2tzLmZvckVhY2goZnVuY3Rpb24gKGV2ZW50VGFzaykge1xuICAgICAgICAgICAgICAgICAgICAvLyB3aW5kb3dzIGhhcyBhZGRlZCB1bmhhbmRsZWRyZWplY3Rpb24gZXZlbnQgbGlzdGVuZXJcbiAgICAgICAgICAgICAgICAgICAgLy8gdHJpZ2dlciB0aGUgZXZlbnQgbGlzdGVuZXJcbiAgICAgICAgICAgICAgICAgICAgdmFyIFByb21pc2VSZWplY3Rpb25FdmVudCA9IGdsb2JhbFsnUHJvbWlzZVJlamVjdGlvbkV2ZW50J107XG4gICAgICAgICAgICAgICAgICAgIGlmIChQcm9taXNlUmVqZWN0aW9uRXZlbnQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBldnQgPSBuZXcgUHJvbWlzZVJlamVjdGlvbkV2ZW50KGV2dE5hbWUsIHsgcHJvbWlzZTogZS5wcm9taXNlLCByZWFzb246IGUucmVqZWN0aW9uIH0pO1xuICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnRUYXNrLmludm9rZShldnQpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9O1xuICAgICAgICB9XG4gICAgICAgIGlmIChnbG9iYWxbJ1Byb21pc2VSZWplY3Rpb25FdmVudCddKSB7XG4gICAgICAgICAgICBab25lW3pvbmVTeW1ib2wkMSgndW5oYW5kbGVkUHJvbWlzZVJlamVjdGlvbkhhbmRsZXInKV0gPVxuICAgICAgICAgICAgICAgIGZpbmRQcm9taXNlUmVqZWN0aW9uSGFuZGxlcigndW5oYW5kbGVkcmVqZWN0aW9uJyk7XG4gICAgICAgICAgICBab25lW3pvbmVTeW1ib2wkMSgncmVqZWN0aW9uSGFuZGxlZEhhbmRsZXInKV0gPVxuICAgICAgICAgICAgICAgIGZpbmRQcm9taXNlUmVqZWN0aW9uSGFuZGxlcigncmVqZWN0aW9uaGFuZGxlZCcpO1xuICAgICAgICB9XG4gICAgfSk7XG59KSk7XG4iXSwibmFtZXMiOlsidGhpcyIsImdsb2JhbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7O0NBQ0EsSUFBSSxhQUFhLEdBQUcsQ0FBQ0EsY0FBSSxJQUFJQSxjQUFJLENBQUMsYUFBYSxLQUFLLFVBQVUsRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUU7Q0FDOUUsSUFBSSxJQUFJLElBQUksSUFBSSxTQUFTLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRSxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxFQUFFLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtDQUN6RixRQUFRLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxJQUFJLElBQUksQ0FBQyxFQUFFO0NBQ2hD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
