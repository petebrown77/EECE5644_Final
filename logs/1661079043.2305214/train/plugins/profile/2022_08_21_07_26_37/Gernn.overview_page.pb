?	NA~6r?@NA~6r?@!NA~6r?@	H/??@H/??@!H/??@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'NA~6r?@1???A??I??t?@Y??????r0*/?Z
|@)      0=2J
Iterator::Root::MapN??1?M??!D?W>c?O@)?8?#+???1????IG@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2A׾?^???!NZ?
{+1@)A׾?^???1NZ?
{+1@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA?,_????!ZQ?$_O1@)C7?嶭?1mjk??)@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatT?^P??!??;??&@) ??X??1??K???!@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicej?t???!?p??>@)j?t???1?p??>@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??????!??*U?A@)-σ??v??1???q?@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??5"??!?????@)??5"??1?????@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?F??1???!?8?
?3@)??4c?t??1;/?W?@:Preprocessing2E
Iterator::Root????????!??jU.yP@)l
dv???1ؿ?ǖ/@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?80.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9H/??@I????T@QKO???/@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	???A?????A??!???A??*      ??!       2      ??!       :	??t?@??t?@!??t?@B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JGPUYH/??@b q????T@yKO???/@?"7
sequential/dense_1/MatMulMatMulP?/?!P?/?0"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum\/e?zx??!V???F???"[
+SGD/SGD/update_8/ResourceApplyKerasMomentumResourceApplyKerasMomentum\/e?zx??!,}T?ٛ?"J
,gradient_tape/sequential/dense/MatMul/MatMulMatMul\/e?zx??!???????0"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul?|??f??!??f$??0"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumc????S??!?sD?Q???"[
+SGD/SGD/update_9/ResourceApplyKerasMomentumResourceApplyKerasMomentumc????S??!!3??'??"L
0gradient_tape/sequential/dense_3/MatMul/MatMul_1MatMulc????S??!m,??q??"L
0gradient_tape/sequential/dense_2/MatMul/MatMul_1MatMulie?t?.??!y5?p???"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentumo 5K
??!(??????Q      Y@Ys?3R1?@a;#s?3Q@q????O?U@y5O?D???"?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?80.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?86.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 