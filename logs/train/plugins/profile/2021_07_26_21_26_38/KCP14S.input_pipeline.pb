	I??&??I??&??!I??&??	??HJ?@??HJ?@!??HJ?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:I??&?????9#J??Ay?&1???Y??K7?A??rEagerKernelExecute 0*	??????e@2U
Iterator::Model::ParallelMapV2p_?Q??!P鮁bc=@)p_?Q??1P鮁bc=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ZӼ???!??.??<@@)?3??7??1??;E;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~j?t??!??,	?5@)8gDio??1????tZ2@:Preprocessing2F
Iterator::Model???N@??!Ĵ??jC@)K?=?U??1p ???~!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?\m?????!<K??N@)??ZӼ???1A????S@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??~j?t??!??,	?@)??~j?t??1??,	?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??0?*x?!6?_-??
@)??0?*x?16?_-??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!?#Y?'A@)-C??6j?1??IeF??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??HJ?@Ir"[?UW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???9#J?????9#J??!???9#J??      ??!       "      ??!       *      ??!       2	y?&1???y?&1???!y?&1???:      ??!       B      ??!       J	??K7?A????K7?A??!??K7?A??R      ??!       Z	??K7?A????K7?A??!??K7?A??b      ??!       JCPU_ONLYY??HJ?@b qr"[?UW@