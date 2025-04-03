// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_AUDIO_OPS_H_
#define TENSORFLOW_CC_OPS_AUDIO_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup audio_ops Audio Ops
/// @{

/// Produces a visualization of audio data over time.
///
/// Spectrograms are a standard way of representing audio information as a series of
/// slices of frequency information, one slice for each window of time. By joining
/// these together into a sequence, they form a distinctive fingerprint of the sound
/// over time.
///
/// This op expects to receive audio data as an input, stored as floats in the range
/// -1 to 1, together with a window width in samples, and a stride specifying how
/// far to move the window between slices. From this it generates a three
/// dimensional output. The first dimension is for the channels in the input, so a
/// stereo audio input would have two here for example. The second dimension is time,
/// with successive frequency slices. The third dimension has an amplitude value for
/// each frequency during that time slice.
///
/// This means the layout when converted and saved as an image is rotated 90 degrees
/// clockwise from a typical spectrogram. Time is descending down the Y axis, and
/// the frequency decreases from left to right.
///
/// Each value in the result represents the square root of the sum of the real and
/// imaginary parts of an FFT on the current window of samples. In this way, the
/// lowest dimension represents the power of each frequency in the current window,
/// and adjacent windows are concatenated in the next dimension.
///
/// To get a more intuitive and visual look at what this operation does, you can run
/// tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
/// resulting spectrogram as a PNG image.
///
/// Args:
/// * scope: A Scope object
/// * input: Float representation of audio data.
/// * window_size: How wide the input window is in samples. For the highest efficiency
/// this should be a power of two, but other values are accepted.
/// * stride: How widely apart the center of adjacent sample windows should be.
///
/// Optional attributes (see `Attrs`):
/// * magnitude_squared: Whether to return the squared magnitude or just the
/// magnitude. Using squared magnitude can avoid extra calculations.
///
/// Returns:
/// * `Output`: 3D representation of the audio frequencies as an image.
class AudioSpectrogram {
 public:
  /// Optional attribute setters for AudioSpectrogram
  struct Attrs {
    /// Whether to return the squared magnitude or just the
    /// magnitude. Using squared magnitude can avoid extra calculations.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MagnitudeSquared(bool x) {
      Attrs ret = *this;
      ret.magnitude_squared_ = x;
      return ret;
    }

    bool magnitude_squared_ = false;
  };
  AudioSpectrogram(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 int64 window_size, int64 stride);
  AudioSpectrogram(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 int64 window_size, int64 stride, const
                 AudioSpectrogram::Attrs& attrs);
  operator ::tensorflow::Output() const { return spectrogram; }
  operator ::tensorflow::Input() const { return spectrogram; }
  ::tensorflow::Node* node() const { return spectrogram.node(); }

  static Attrs MagnitudeSquared(bool x) {
    return Attrs().MagnitudeSquared(x);
  }

  Operation operation;
  ::tensorflow::Output spectrogram;
};

/// Decode a 16-bit PCM WAV file to a float tensor.
///
/// The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
///
/// When desired_channels is set, if the input contains fewer channels than this
/// then the last channel will be duplicated to give the requested number, else if
/// the input has more channels than requested then the additional channels will be
/// ignored.
///
/// If desired_samples is set, then the audio will be cropped or padded with zeroes
/// to the requested length.
///
/// The first output contains a Tensor with the content of the audio samples. The
/// lowest dimension will be the number of channels, and the second will be the
/// number of samples. For example, a ten-sample-long stereo WAV file should give an
/// output shape of [10, 2].
///
/// Args:
/// * scope: A Scope object
/// * contents: The WAV-encoded audio, usually from a file.
///
/// Optional attributes (see `Attrs`):
/// * desired_channels: Number of sample channels wanted.
/// * desired_samples: Length of audio requested.
///
/// Returns:
/// * `Output` audio: 2-D with shape `[length, channels]`.
/// * `Output` sample_rate: Scalar holding the sample rate found in the WAV header.
class DecodeWav {
 public:
  /// Optional attribute setters for DecodeWav
  struct Attrs {
    /// Number of sample channels wanted.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs DesiredChannels(int64 x) {
      Attrs ret = *this;
      ret.desired_channels_ = x;
      return ret;
    }

    /// Length of audio requested.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs DesiredSamples(int64 x) {
      Attrs ret = *this;
      ret.desired_samples_ = x;
      return ret;
    }

    int64 desired_channels_ = -1;
    int64 desired_samples_ = -1;
  };
  DecodeWav(const ::tensorflow::Scope& scope, ::tensorflow::Input contents);
  DecodeWav(const ::tensorflow::Scope& scope, ::tensorflow::Input contents, const
          DecodeWav::Attrs& attrs);

  static Attrs DesiredChannels(int64 x) {
    return Attrs().DesiredChannels(x);
  }
  static Attrs DesiredSamples(int64 x) {
    return Attrs().DesiredSamples(x);
  }

  Operation operation;
  ::tensorflow::Output audio;
  ::tensorflow::Output sample_rate;
};

/// Encode audio data using the WAV file format.
///
/// This operation will generate a string suitable to be saved out to create a .wav
/// audio file. It will be encoded in the 16-bit PCM format. It takes in float
/// values in the range -1.0f to 1.0f, and any outside that value will be clamped to
/// that range.
///
/// `audio` is a 2-D float Tensor of shape `[length, channels]`.
/// `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).
///
/// Args:
/// * scope: A Scope object
/// * audio: 2-D with shape `[length, channels]`.
/// * sample_rate: Scalar containing the sample frequency.
///
/// Returns:
/// * `Output`: 0-D. WAV-encoded file contents.
class EncodeWav {
 public:
  EncodeWav(const ::tensorflow::Scope& scope, ::tensorflow::Input audio,
          ::tensorflow::Input sample_rate);
  operator ::tensorflow::Output() const { return contents; }
  operator ::tensorflow::Input() const { return contents; }
  ::tensorflow::Node* node() const { return contents.node(); }

  Operation operation;
  ::tensorflow::Output contents;
};

/// Transforms a spectrogram into a form that's useful for speech recognition.
///
/// Mel Frequency Cepstral Coefficients are a way of representing audio data that's
/// been effective as an input feature for machine learning. They are created by
/// taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
/// higher frequencies that are less significant to the human ear. They have a long
/// history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
/// is a good resource to learn more.
///
/// Args:
/// * scope: A Scope object
/// * spectrogram: Typically produced by the Spectrogram op, with magnitude_squared
/// set to true.
/// * sample_rate: How many samples per second the source audio used.
///
/// Optional attributes (see `Attrs`):
/// * upper_frequency_limit: The highest frequency to use when calculating the
/// ceptstrum.
/// * lower_frequency_limit: The lowest frequency to use when calculating the
/// ceptstrum.
/// * filterbank_channel_count: Resolution of the Mel bank used internally.
/// * dct_coefficient_count: How many output channels to produce per time slice.
///
/// Returns:
/// * `Output`: The output tensor.
class Mfcc {
 public:
  /// Optional attribute setters for Mfcc
  struct Attrs {
    /// The highest frequency to use when calculating the
    /// ceptstrum.
    ///
    /// Defaults to 4000
    TF_MUST_USE_RESULT Attrs UpperFrequencyLimit(float x) {
      Attrs ret = *this;
      ret.upper_frequency_limit_ = x;
      return ret;
    }

    /// The lowest frequency to use when calculating the
    /// ceptstrum.
    ///
    /// Defaults to 20
    TF_MUST_USE_RESULT Attrs LowerFrequencyLimit(float x) {
      Attrs ret = *this;
      ret.lower_frequency_limit_ = x;
      return ret;
    }

    /// Resolution of the Mel bank used internally.
    ///
    /// Defaults to 40
    TF_MUST_USE_RESULT Attrs FilterbankChannelCount(int64 x) {
      Attrs ret = *this;
      ret.filterbank_channel_count_ = x;
      return ret;
    }

    /// How many output channels to produce per time slice.
    ///
    /// Defaults to 13
    TF_MUST_USE_RESULT Attrs DctCoefficientCount(int64 x) {
      Attrs ret = *this;
      ret.dct_coefficient_count_ = x;
      return ret;
    }

    float upper_frequency_limit_ = 4000.0f;
    float lower_frequency_limit_ = 20.0f;
    int64 filterbank_channel_count_ = 40;
    int64 dct_coefficient_count_ = 13;
  };
  Mfcc(const ::tensorflow::Scope& scope, ::tensorflow::Input spectrogram,
     ::tensorflow::Input sample_rate);
  Mfcc(const ::tensorflow::Scope& scope, ::tensorflow::Input spectrogram,
     ::tensorflow::Input sample_rate, const Mfcc::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs UpperFrequencyLimit(float x) {
    return Attrs().UpperFrequencyLimit(x);
  }
  static Attrs LowerFrequencyLimit(float x) {
    return Attrs().LowerFrequencyLimit(x);
  }
  static Attrs FilterbankChannelCount(int64 x) {
    return Attrs().FilterbankChannelCount(x);
  }
  static Attrs DctCoefficientCount(int64 x) {
    return Attrs().DctCoefficientCount(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_AUDIO_OPS_H_
