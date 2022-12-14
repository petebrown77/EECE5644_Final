?	?x??G@?x??G@!?x??G@	?&??S@@?&??S@@!?&??S@@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?x??G@1%!???'??I<????@Y?=?
Y??r0*	??? ???@2J
Iterator::Root::Map?????K??!~????P@)?? @???1?7'??G@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?w???!?Q?}?t3@)?w???1?Q?}?t3@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat??DKO??!Y???+@)?F?ү?1ٗ??&@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?(??{??!?=o/(?'@)??5"??1v??O? @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?5?????! ?iqa?@)?5?????1 ?iqa?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipr???_??!?F'?|?>@)?W?B???1??n???@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor>?$@M-??!?oE?@)>?$@M-??1?oE?@:Preprocessing2E
Iterator::Root???S??!Y.6??RQ@)e??Q??1K{??{?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap2????s??!??*???+@)?z?V????1 ???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?&??S@@I??????S@Q? !#?1@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	%!???'??%!???'??!%!???'??*      ??!       2      ??!       :	<????@<????@!<????@B      ??!       J	?=?
Y???=?
Y??!?=?
Y??R      ??!       Z	?=?
Y???=?
Y??!?=?
Y??b      ??!       JGPUY?&??S@@b q??????S@y? !#?1@?"7
sequential/dense_2/MatMulMatMul???_??!???_??0"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum
+gk(??!h?D??"7
sequential/dense_3/MatMulMatMul
+gk(??!?.?hLX??0"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMulC?????!("?-??0"L
.gradient_tape/sequential/dense_3/MatMul/MatMulMatMulOl????!?;?&M???0"L
0gradient_tape/sequential/dense_3/MatMul/MatMul_1MatMulOl????!?OXY?%??"7
sequential/dense_1/MatMulMatMulOl????!Zc??ס??0"[
+SGD/SGD/update_9/ResourceApplyKerasMomentumResourceApplyKerasMomentum[4??ށ?!=????"L
.gradient_tape/sequential/dense_2/MatMul/MatMulMatMul[4??ށ?!uȆk?H??0"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum(s?-????!?V?1???Q      Y@Ys?3R1?@a;#s?3Q@q?[Ke^kU@ysI?????"?
device?Your program is NOT input-bound because only 3.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?78.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?85.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 