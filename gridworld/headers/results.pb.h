// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: results.proto

#ifndef PROTOBUF_results_2eproto__INCLUDED
#define PROTOBUF_results_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace evaluation {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_results_2eproto();
void protobuf_AssignDesc_results_2eproto();
void protobuf_ShutdownFile_results_2eproto();

class ImprovementResults;
class MethodResult;

// ===================================================================

class MethodResult : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:evaluation.MethodResult) */ {
 public:
  MethodResult();
  virtual ~MethodResult();

  MethodResult(const MethodResult& from);

  inline MethodResult& operator=(const MethodResult& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const MethodResult& default_instance();

  void Swap(MethodResult* other);

  // implements Message ----------------------------------------------

  inline MethodResult* New() const { return New(NULL); }

  MethodResult* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const MethodResult& from);
  void MergeFrom(const MethodResult& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(MethodResult* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required string method_name = 1;
  bool has_method_name() const;
  void clear_method_name();
  static const int kMethodNameFieldNumber = 1;
  const ::std::string& method_name() const;
  void set_method_name(const ::std::string& value);
  void set_method_name(const char* value);
  void set_method_name(const char* value, size_t size);
  ::std::string* mutable_method_name();
  ::std::string* release_method_name();
  void set_allocated_method_name(::std::string* method_name);

  // repeated float num_steps_observed = 2;
  int num_steps_observed_size() const;
  void clear_num_steps_observed();
  static const int kNumStepsObservedFieldNumber = 2;
  float num_steps_observed(int index) const;
  void set_num_steps_observed(int index, float value);
  void add_num_steps_observed(float value);
  const ::google::protobuf::RepeatedField< float >&
      num_steps_observed() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_num_steps_observed();

  // repeated float value_error = 3;
  int value_error_size() const;
  void clear_value_error();
  static const int kValueErrorFieldNumber = 3;
  float value_error(int index) const;
  void set_value_error(int index, float value);
  void add_value_error(float value);
  const ::google::protobuf::RepeatedField< float >&
      value_error() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_value_error();

  // repeated float num_unvisited_s_a = 4;
  int num_unvisited_s_a_size() const;
  void clear_num_unvisited_s_a();
  static const int kNumUnvisitedSAFieldNumber = 4;
  float num_unvisited_s_a(int index) const;
  void set_num_unvisited_s_a(int index, float value);
  void add_num_unvisited_s_a(float value);
  const ::google::protobuf::RepeatedField< float >&
      num_unvisited_s_a() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_num_unvisited_s_a();

  // repeated float deterministic_prob = 5;
  int deterministic_prob_size() const;
  void clear_deterministic_prob();
  static const int kDeterministicProbFieldNumber = 5;
  float deterministic_prob(int index) const;
  void set_deterministic_prob(int index, float value);
  void add_deterministic_prob(float value);
  const ::google::protobuf::RepeatedField< float >&
      deterministic_prob() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_deterministic_prob();

  // repeated float batch_process_num_steps = 6;
  int batch_process_num_steps_size() const;
  void clear_batch_process_num_steps();
  static const int kBatchProcessNumStepsFieldNumber = 6;
  float batch_process_num_steps(int index) const;
  void set_batch_process_num_steps(int index, float value);
  void add_batch_process_num_steps(float value);
  const ::google::protobuf::RepeatedField< float >&
      batch_process_num_steps() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_batch_process_num_steps();

  // repeated float batch_process_mses = 7;
  int batch_process_mses_size() const;
  void clear_batch_process_mses();
  static const int kBatchProcessMsesFieldNumber = 7;
  float batch_process_mses(int index) const;
  void set_batch_process_mses(int index, float value);
  void add_batch_process_mses(float value);
  const ::google::protobuf::RepeatedField< float >&
      batch_process_mses() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_batch_process_mses();

  // @@protoc_insertion_point(class_scope:evaluation.MethodResult)
 private:
  inline void set_has_method_name();
  inline void clear_has_method_name();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::internal::ArenaStringPtr method_name_;
  ::google::protobuf::RepeatedField< float > num_steps_observed_;
  ::google::protobuf::RepeatedField< float > value_error_;
  ::google::protobuf::RepeatedField< float > num_unvisited_s_a_;
  ::google::protobuf::RepeatedField< float > deterministic_prob_;
  ::google::protobuf::RepeatedField< float > batch_process_num_steps_;
  ::google::protobuf::RepeatedField< float > batch_process_mses_;
  friend void  protobuf_AddDesc_results_2eproto();
  friend void protobuf_AssignDesc_results_2eproto();
  friend void protobuf_ShutdownFile_results_2eproto();

  void InitAsDefaultInstance();
  static MethodResult* default_instance_;
};
// -------------------------------------------------------------------

class ImprovementResults : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:evaluation.ImprovementResults) */ {
 public:
  ImprovementResults();
  virtual ~ImprovementResults();

  ImprovementResults(const ImprovementResults& from);

  inline ImprovementResults& operator=(const ImprovementResults& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const ImprovementResults& default_instance();

  void Swap(ImprovementResults* other);

  // implements Message ----------------------------------------------

  inline ImprovementResults* New() const { return New(NULL); }

  ImprovementResults* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const ImprovementResults& from);
  void MergeFrom(const ImprovementResults& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(ImprovementResults* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated float dataset_sizes = 1;
  int dataset_sizes_size() const;
  void clear_dataset_sizes();
  static const int kDatasetSizesFieldNumber = 1;
  float dataset_sizes(int index) const;
  void set_dataset_sizes(int index, float value);
  void add_dataset_sizes(float value);
  const ::google::protobuf::RepeatedField< float >&
      dataset_sizes() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_dataset_sizes();

  // repeated float avg_return = 2;
  int avg_return_size() const;
  void clear_avg_return();
  static const int kAvgReturnFieldNumber = 2;
  float avg_return(int index) const;
  void set_avg_return(int index, float value);
  void add_avg_return(float value);
  const ::google::protobuf::RepeatedField< float >&
      avg_return() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_avg_return();

  // repeated float estimated_avg_return = 3;
  int estimated_avg_return_size() const;
  void clear_estimated_avg_return();
  static const int kEstimatedAvgReturnFieldNumber = 3;
  float estimated_avg_return(int index) const;
  void set_estimated_avg_return(int index, float value);
  void add_estimated_avg_return(float value);
  const ::google::protobuf::RepeatedField< float >&
      estimated_avg_return() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_estimated_avg_return();

  // repeated float mse = 4;
  int mse_size() const;
  void clear_mse();
  static const int kMseFieldNumber = 4;
  float mse(int index) const;
  void set_mse(int index, float value);
  void add_mse(float value);
  const ::google::protobuf::RepeatedField< float >&
      mse() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_mse();

  // optional string label = 5;
  bool has_label() const;
  void clear_label();
  static const int kLabelFieldNumber = 5;
  const ::std::string& label() const;
  void set_label(const ::std::string& value);
  void set_label(const char* value);
  void set_label(const char* value, size_t size);
  ::std::string* mutable_label();
  ::std::string* release_label();
  void set_allocated_label(::std::string* label);

  // @@protoc_insertion_point(class_scope:evaluation.ImprovementResults)
 private:
  inline void set_has_label();
  inline void clear_has_label();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< float > dataset_sizes_;
  ::google::protobuf::RepeatedField< float > avg_return_;
  ::google::protobuf::RepeatedField< float > estimated_avg_return_;
  ::google::protobuf::RepeatedField< float > mse_;
  ::google::protobuf::internal::ArenaStringPtr label_;
  friend void  protobuf_AddDesc_results_2eproto();
  friend void protobuf_AssignDesc_results_2eproto();
  friend void protobuf_ShutdownFile_results_2eproto();

  void InitAsDefaultInstance();
  static ImprovementResults* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// MethodResult

// required string method_name = 1;
inline bool MethodResult::has_method_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void MethodResult::set_has_method_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void MethodResult::clear_has_method_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void MethodResult::clear_method_name() {
  method_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_method_name();
}
inline const ::std::string& MethodResult::method_name() const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.method_name)
  return method_name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void MethodResult::set_method_name(const ::std::string& value) {
  set_has_method_name();
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.method_name)
}
inline void MethodResult::set_method_name(const char* value) {
  set_has_method_name();
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:evaluation.MethodResult.method_name)
}
inline void MethodResult::set_method_name(const char* value, size_t size) {
  set_has_method_name();
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:evaluation.MethodResult.method_name)
}
inline ::std::string* MethodResult::mutable_method_name() {
  set_has_method_name();
  // @@protoc_insertion_point(field_mutable:evaluation.MethodResult.method_name)
  return method_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* MethodResult::release_method_name() {
  // @@protoc_insertion_point(field_release:evaluation.MethodResult.method_name)
  clear_has_method_name();
  return method_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void MethodResult::set_allocated_method_name(::std::string* method_name) {
  if (method_name != NULL) {
    set_has_method_name();
  } else {
    clear_has_method_name();
  }
  method_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), method_name);
  // @@protoc_insertion_point(field_set_allocated:evaluation.MethodResult.method_name)
}

// repeated float num_steps_observed = 2;
inline int MethodResult::num_steps_observed_size() const {
  return num_steps_observed_.size();
}
inline void MethodResult::clear_num_steps_observed() {
  num_steps_observed_.Clear();
}
inline float MethodResult::num_steps_observed(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.num_steps_observed)
  return num_steps_observed_.Get(index);
}
inline void MethodResult::set_num_steps_observed(int index, float value) {
  num_steps_observed_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.num_steps_observed)
}
inline void MethodResult::add_num_steps_observed(float value) {
  num_steps_observed_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.num_steps_observed)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::num_steps_observed() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.num_steps_observed)
  return num_steps_observed_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_num_steps_observed() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.num_steps_observed)
  return &num_steps_observed_;
}

// repeated float value_error = 3;
inline int MethodResult::value_error_size() const {
  return value_error_.size();
}
inline void MethodResult::clear_value_error() {
  value_error_.Clear();
}
inline float MethodResult::value_error(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.value_error)
  return value_error_.Get(index);
}
inline void MethodResult::set_value_error(int index, float value) {
  value_error_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.value_error)
}
inline void MethodResult::add_value_error(float value) {
  value_error_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.value_error)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::value_error() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.value_error)
  return value_error_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_value_error() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.value_error)
  return &value_error_;
}

// repeated float num_unvisited_s_a = 4;
inline int MethodResult::num_unvisited_s_a_size() const {
  return num_unvisited_s_a_.size();
}
inline void MethodResult::clear_num_unvisited_s_a() {
  num_unvisited_s_a_.Clear();
}
inline float MethodResult::num_unvisited_s_a(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.num_unvisited_s_a)
  return num_unvisited_s_a_.Get(index);
}
inline void MethodResult::set_num_unvisited_s_a(int index, float value) {
  num_unvisited_s_a_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.num_unvisited_s_a)
}
inline void MethodResult::add_num_unvisited_s_a(float value) {
  num_unvisited_s_a_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.num_unvisited_s_a)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::num_unvisited_s_a() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.num_unvisited_s_a)
  return num_unvisited_s_a_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_num_unvisited_s_a() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.num_unvisited_s_a)
  return &num_unvisited_s_a_;
}

// repeated float deterministic_prob = 5;
inline int MethodResult::deterministic_prob_size() const {
  return deterministic_prob_.size();
}
inline void MethodResult::clear_deterministic_prob() {
  deterministic_prob_.Clear();
}
inline float MethodResult::deterministic_prob(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.deterministic_prob)
  return deterministic_prob_.Get(index);
}
inline void MethodResult::set_deterministic_prob(int index, float value) {
  deterministic_prob_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.deterministic_prob)
}
inline void MethodResult::add_deterministic_prob(float value) {
  deterministic_prob_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.deterministic_prob)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::deterministic_prob() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.deterministic_prob)
  return deterministic_prob_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_deterministic_prob() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.deterministic_prob)
  return &deterministic_prob_;
}

// repeated float batch_process_num_steps = 6;
inline int MethodResult::batch_process_num_steps_size() const {
  return batch_process_num_steps_.size();
}
inline void MethodResult::clear_batch_process_num_steps() {
  batch_process_num_steps_.Clear();
}
inline float MethodResult::batch_process_num_steps(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.batch_process_num_steps)
  return batch_process_num_steps_.Get(index);
}
inline void MethodResult::set_batch_process_num_steps(int index, float value) {
  batch_process_num_steps_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.batch_process_num_steps)
}
inline void MethodResult::add_batch_process_num_steps(float value) {
  batch_process_num_steps_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.batch_process_num_steps)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::batch_process_num_steps() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.batch_process_num_steps)
  return batch_process_num_steps_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_batch_process_num_steps() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.batch_process_num_steps)
  return &batch_process_num_steps_;
}

// repeated float batch_process_mses = 7;
inline int MethodResult::batch_process_mses_size() const {
  return batch_process_mses_.size();
}
inline void MethodResult::clear_batch_process_mses() {
  batch_process_mses_.Clear();
}
inline float MethodResult::batch_process_mses(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.MethodResult.batch_process_mses)
  return batch_process_mses_.Get(index);
}
inline void MethodResult::set_batch_process_mses(int index, float value) {
  batch_process_mses_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.MethodResult.batch_process_mses)
}
inline void MethodResult::add_batch_process_mses(float value) {
  batch_process_mses_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.MethodResult.batch_process_mses)
}
inline const ::google::protobuf::RepeatedField< float >&
MethodResult::batch_process_mses() const {
  // @@protoc_insertion_point(field_list:evaluation.MethodResult.batch_process_mses)
  return batch_process_mses_;
}
inline ::google::protobuf::RepeatedField< float >*
MethodResult::mutable_batch_process_mses() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.MethodResult.batch_process_mses)
  return &batch_process_mses_;
}

// -------------------------------------------------------------------

// ImprovementResults

// repeated float dataset_sizes = 1;
inline int ImprovementResults::dataset_sizes_size() const {
  return dataset_sizes_.size();
}
inline void ImprovementResults::clear_dataset_sizes() {
  dataset_sizes_.Clear();
}
inline float ImprovementResults::dataset_sizes(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.ImprovementResults.dataset_sizes)
  return dataset_sizes_.Get(index);
}
inline void ImprovementResults::set_dataset_sizes(int index, float value) {
  dataset_sizes_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.ImprovementResults.dataset_sizes)
}
inline void ImprovementResults::add_dataset_sizes(float value) {
  dataset_sizes_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.ImprovementResults.dataset_sizes)
}
inline const ::google::protobuf::RepeatedField< float >&
ImprovementResults::dataset_sizes() const {
  // @@protoc_insertion_point(field_list:evaluation.ImprovementResults.dataset_sizes)
  return dataset_sizes_;
}
inline ::google::protobuf::RepeatedField< float >*
ImprovementResults::mutable_dataset_sizes() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.ImprovementResults.dataset_sizes)
  return &dataset_sizes_;
}

// repeated float avg_return = 2;
inline int ImprovementResults::avg_return_size() const {
  return avg_return_.size();
}
inline void ImprovementResults::clear_avg_return() {
  avg_return_.Clear();
}
inline float ImprovementResults::avg_return(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.ImprovementResults.avg_return)
  return avg_return_.Get(index);
}
inline void ImprovementResults::set_avg_return(int index, float value) {
  avg_return_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.ImprovementResults.avg_return)
}
inline void ImprovementResults::add_avg_return(float value) {
  avg_return_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.ImprovementResults.avg_return)
}
inline const ::google::protobuf::RepeatedField< float >&
ImprovementResults::avg_return() const {
  // @@protoc_insertion_point(field_list:evaluation.ImprovementResults.avg_return)
  return avg_return_;
}
inline ::google::protobuf::RepeatedField< float >*
ImprovementResults::mutable_avg_return() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.ImprovementResults.avg_return)
  return &avg_return_;
}

// repeated float estimated_avg_return = 3;
inline int ImprovementResults::estimated_avg_return_size() const {
  return estimated_avg_return_.size();
}
inline void ImprovementResults::clear_estimated_avg_return() {
  estimated_avg_return_.Clear();
}
inline float ImprovementResults::estimated_avg_return(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.ImprovementResults.estimated_avg_return)
  return estimated_avg_return_.Get(index);
}
inline void ImprovementResults::set_estimated_avg_return(int index, float value) {
  estimated_avg_return_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.ImprovementResults.estimated_avg_return)
}
inline void ImprovementResults::add_estimated_avg_return(float value) {
  estimated_avg_return_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.ImprovementResults.estimated_avg_return)
}
inline const ::google::protobuf::RepeatedField< float >&
ImprovementResults::estimated_avg_return() const {
  // @@protoc_insertion_point(field_list:evaluation.ImprovementResults.estimated_avg_return)
  return estimated_avg_return_;
}
inline ::google::protobuf::RepeatedField< float >*
ImprovementResults::mutable_estimated_avg_return() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.ImprovementResults.estimated_avg_return)
  return &estimated_avg_return_;
}

// repeated float mse = 4;
inline int ImprovementResults::mse_size() const {
  return mse_.size();
}
inline void ImprovementResults::clear_mse() {
  mse_.Clear();
}
inline float ImprovementResults::mse(int index) const {
  // @@protoc_insertion_point(field_get:evaluation.ImprovementResults.mse)
  return mse_.Get(index);
}
inline void ImprovementResults::set_mse(int index, float value) {
  mse_.Set(index, value);
  // @@protoc_insertion_point(field_set:evaluation.ImprovementResults.mse)
}
inline void ImprovementResults::add_mse(float value) {
  mse_.Add(value);
  // @@protoc_insertion_point(field_add:evaluation.ImprovementResults.mse)
}
inline const ::google::protobuf::RepeatedField< float >&
ImprovementResults::mse() const {
  // @@protoc_insertion_point(field_list:evaluation.ImprovementResults.mse)
  return mse_;
}
inline ::google::protobuf::RepeatedField< float >*
ImprovementResults::mutable_mse() {
  // @@protoc_insertion_point(field_mutable_list:evaluation.ImprovementResults.mse)
  return &mse_;
}

// optional string label = 5;
inline bool ImprovementResults::has_label() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void ImprovementResults::set_has_label() {
  _has_bits_[0] |= 0x00000010u;
}
inline void ImprovementResults::clear_has_label() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void ImprovementResults::clear_label() {
  label_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_label();
}
inline const ::std::string& ImprovementResults::label() const {
  // @@protoc_insertion_point(field_get:evaluation.ImprovementResults.label)
  return label_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void ImprovementResults::set_label(const ::std::string& value) {
  set_has_label();
  label_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:evaluation.ImprovementResults.label)
}
inline void ImprovementResults::set_label(const char* value) {
  set_has_label();
  label_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:evaluation.ImprovementResults.label)
}
inline void ImprovementResults::set_label(const char* value, size_t size) {
  set_has_label();
  label_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:evaluation.ImprovementResults.label)
}
inline ::std::string* ImprovementResults::mutable_label() {
  set_has_label();
  // @@protoc_insertion_point(field_mutable:evaluation.ImprovementResults.label)
  return label_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* ImprovementResults::release_label() {
  // @@protoc_insertion_point(field_release:evaluation.ImprovementResults.label)
  clear_has_label();
  return label_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void ImprovementResults::set_allocated_label(::std::string* label) {
  if (label != NULL) {
    set_has_label();
  } else {
    clear_has_label();
  }
  label_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), label);
  // @@protoc_insertion_point(field_set_allocated:evaluation.ImprovementResults.label)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace evaluation

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_results_2eproto__INCLUDED
