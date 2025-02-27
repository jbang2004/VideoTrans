# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cosyvoice.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x63osyvoice.proto\x12\tcosyvoice\"$\n\x14NormalizeTextRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"*\n\x15NormalizeTextResponse\x12\x11\n\ttext_uuid\x18\x01 \x01(\t\"Y\n\x1d\x45xtractSpeakerFeaturesRequest\x12\x14\n\x0cspeaker_uuid\x18\x01 \x01(\t\x12\r\n\x05\x61udio\x18\x02 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x03 \x01(\x05\"1\n\x1e\x45xtractSpeakerFeaturesResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"C\n\x18GenerateTTSTokensRequest\x12\x11\n\ttext_uuid\x18\x01 \x01(\t\x12\x14\n\x0cspeaker_uuid\x18\x02 \x01(\t\"A\n\x19GenerateTTSTokensResponse\x12\x13\n\x0b\x64uration_ms\x18\x01 \x01(\x05\x12\x0f\n\x07success\x18\x02 \x01(\x08\"J\n\x10Token2WavRequest\x12\x11\n\ttext_uuid\x18\x01 \x01(\t\x12\x14\n\x0cspeaker_uuid\x18\x02 \x01(\t\x12\r\n\x05speed\x18\x03 \x01(\x02\"8\n\x11Token2WavResponse\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x14\n\x0c\x64uration_sec\x18\x02 \x01(\x02\"2\n\x0e\x43leanupRequest\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x12\n\nis_speaker\x18\x02 \x01(\x08\"\"\n\x0f\x43leanupResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x32\xbf\x03\n\x10\x43osyVoiceService\x12R\n\rNormalizeText\x12\x1f.cosyvoice.NormalizeTextRequest\x1a .cosyvoice.NormalizeTextResponse\x12m\n\x16\x45xtractSpeakerFeatures\x12(.cosyvoice.ExtractSpeakerFeaturesRequest\x1a).cosyvoice.ExtractSpeakerFeaturesResponse\x12^\n\x11GenerateTTSTokens\x12#.cosyvoice.GenerateTTSTokensRequest\x1a$.cosyvoice.GenerateTTSTokensResponse\x12\x46\n\tToken2Wav\x12\x1b.cosyvoice.Token2WavRequest\x1a\x1c.cosyvoice.Token2WavResponse\x12@\n\x07\x43leanup\x12\x19.cosyvoice.CleanupRequest\x1a\x1a.cosyvoice.CleanupResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cosyvoice_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_NORMALIZETEXTREQUEST']._serialized_start=30
  _globals['_NORMALIZETEXTREQUEST']._serialized_end=66
  _globals['_NORMALIZETEXTRESPONSE']._serialized_start=68
  _globals['_NORMALIZETEXTRESPONSE']._serialized_end=110
  _globals['_EXTRACTSPEAKERFEATURESREQUEST']._serialized_start=112
  _globals['_EXTRACTSPEAKERFEATURESREQUEST']._serialized_end=201
  _globals['_EXTRACTSPEAKERFEATURESRESPONSE']._serialized_start=203
  _globals['_EXTRACTSPEAKERFEATURESRESPONSE']._serialized_end=252
  _globals['_GENERATETTSTOKENSREQUEST']._serialized_start=254
  _globals['_GENERATETTSTOKENSREQUEST']._serialized_end=321
  _globals['_GENERATETTSTOKENSRESPONSE']._serialized_start=323
  _globals['_GENERATETTSTOKENSRESPONSE']._serialized_end=388
  _globals['_TOKEN2WAVREQUEST']._serialized_start=390
  _globals['_TOKEN2WAVREQUEST']._serialized_end=464
  _globals['_TOKEN2WAVRESPONSE']._serialized_start=466
  _globals['_TOKEN2WAVRESPONSE']._serialized_end=522
  _globals['_CLEANUPREQUEST']._serialized_start=524
  _globals['_CLEANUPREQUEST']._serialized_end=574
  _globals['_CLEANUPRESPONSE']._serialized_start=576
  _globals['_CLEANUPRESPONSE']._serialized_end=610
  _globals['_COSYVOICESERVICE']._serialized_start=613
  _globals['_COSYVOICESERVICE']._serialized_end=1060
# @@protoc_insertion_point(module_scope)
