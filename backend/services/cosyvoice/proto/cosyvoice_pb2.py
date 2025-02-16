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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x63osyvoice.proto\x12\tcosyvoice\"\x97\x01\n\x08\x46\x65\x61tures\x12\x11\n\tembedding\x18\x01 \x03(\x02\x12\x1a\n\x12prompt_speech_feat\x18\x02 \x03(\x02\x12\x1b\n\x13prompt_speech_token\x18\x03 \x03(\x05\x12\x1e\n\x16prompt_speech_feat_len\x18\x04 \x01(\x05\x12\x1f\n\x17prompt_speech_token_len\x18\x05 \x01(\x05\"$\n\x14NormalizeTextRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"\x8c\x01\n\x15NormalizeTextResponse\x12:\n\x08segments\x18\x01 \x03(\x0b\x32(.cosyvoice.NormalizeTextResponse.Segment\x1a\x37\n\x07Segment\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x0e\n\x06tokens\x18\x02 \x03(\x05\x12\x0e\n\x06length\x18\x03 \x01(\x05\"\xf7\x01\n\x18GenerateTTSTokensRequest\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x15\n\rtext_segments\x18\x02 \x03(\t\x12N\n\x11tts_token_context\x18\x03 \x01(\x0b\x32\x33.cosyvoice.GenerateTTSTokensRequest.TTSTokenContext\x1a\x66\n\x0fTTSTokenContext\x12\x13\n\x0bprompt_text\x18\x01 \x03(\x05\x12\x17\n\x0fprompt_text_len\x18\x02 \x01(\x05\x12%\n\x08\x66\x65\x61tures\x18\x03 \x01(\x0b\x32\x13.cosyvoice.Features\"\x99\x01\n\x19GenerateTTSTokensResponse\x12>\n\x08segments\x18\x01 \x03(\x0b\x32,.cosyvoice.GenerateTTSTokensResponse.Segment\x12\x13\n\x0b\x64uration_ms\x18\x02 \x01(\x02\x1a\'\n\x07Segment\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x0e\n\x06tokens\x18\x02 \x03(\x05\"\xd1\x01\n\x10Token2WavRequest\x12\x13\n\x0btokens_list\x18\x01 \x03(\x05\x12\x12\n\nuuids_list\x18\x02 \x03(\t\x12\r\n\x05speed\x18\x03 \x01(\x02\x12\x38\n\x07speaker\x18\x04 \x01(\x0b\x32\'.cosyvoice.Token2WavRequest.SpeakerInfo\x1aK\n\x0bSpeakerInfo\x12\x14\n\x0cprompt_token\x18\x01 \x03(\x05\x12\x13\n\x0bprompt_feat\x18\x02 \x03(\x02\x12\x11\n\tembedding\x18\x03 \x03(\x02\"8\n\x11Token2WavResponse\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x14\n\x0c\x64uration_sec\x18\x02 \x01(\x02\"C\n\x1d\x45xtractSpeakerFeaturesRequest\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\x05\"G\n\x1e\x45xtractSpeakerFeaturesResponse\x12%\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x13.cosyvoice.Features2\xfd\x02\n\x10\x43osyVoiceService\x12R\n\rNormalizeText\x12\x1f.cosyvoice.NormalizeTextRequest\x1a .cosyvoice.NormalizeTextResponse\x12^\n\x11GenerateTTSTokens\x12#.cosyvoice.GenerateTTSTokensRequest\x1a$.cosyvoice.GenerateTTSTokensResponse\x12\x46\n\tToken2Wav\x12\x1b.cosyvoice.Token2WavRequest\x1a\x1c.cosyvoice.Token2WavResponse\x12m\n\x16\x45xtractSpeakerFeatures\x12(.cosyvoice.ExtractSpeakerFeaturesRequest\x1a).cosyvoice.ExtractSpeakerFeaturesResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cosyvoice_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_FEATURES']._serialized_start=31
  _globals['_FEATURES']._serialized_end=182
  _globals['_NORMALIZETEXTREQUEST']._serialized_start=184
  _globals['_NORMALIZETEXTREQUEST']._serialized_end=220
  _globals['_NORMALIZETEXTRESPONSE']._serialized_start=223
  _globals['_NORMALIZETEXTRESPONSE']._serialized_end=363
  _globals['_NORMALIZETEXTRESPONSE_SEGMENT']._serialized_start=308
  _globals['_NORMALIZETEXTRESPONSE_SEGMENT']._serialized_end=363
  _globals['_GENERATETTSTOKENSREQUEST']._serialized_start=366
  _globals['_GENERATETTSTOKENSREQUEST']._serialized_end=613
  _globals['_GENERATETTSTOKENSREQUEST_TTSTOKENCONTEXT']._serialized_start=511
  _globals['_GENERATETTSTOKENSREQUEST_TTSTOKENCONTEXT']._serialized_end=613
  _globals['_GENERATETTSTOKENSRESPONSE']._serialized_start=616
  _globals['_GENERATETTSTOKENSRESPONSE']._serialized_end=769
  _globals['_GENERATETTSTOKENSRESPONSE_SEGMENT']._serialized_start=730
  _globals['_GENERATETTSTOKENSRESPONSE_SEGMENT']._serialized_end=769
  _globals['_TOKEN2WAVREQUEST']._serialized_start=772
  _globals['_TOKEN2WAVREQUEST']._serialized_end=981
  _globals['_TOKEN2WAVREQUEST_SPEAKERINFO']._serialized_start=906
  _globals['_TOKEN2WAVREQUEST_SPEAKERINFO']._serialized_end=981
  _globals['_TOKEN2WAVRESPONSE']._serialized_start=983
  _globals['_TOKEN2WAVRESPONSE']._serialized_end=1039
  _globals['_EXTRACTSPEAKERFEATURESREQUEST']._serialized_start=1041
  _globals['_EXTRACTSPEAKERFEATURESREQUEST']._serialized_end=1108
  _globals['_EXTRACTSPEAKERFEATURESRESPONSE']._serialized_start=1110
  _globals['_EXTRACTSPEAKERFEATURESRESPONSE']._serialized_end=1181
  _globals['_COSYVOICESERVICE']._serialized_start=1184
  _globals['_COSYVOICESERVICE']._serialized_end=1565
# @@protoc_insertion_point(module_scope)
