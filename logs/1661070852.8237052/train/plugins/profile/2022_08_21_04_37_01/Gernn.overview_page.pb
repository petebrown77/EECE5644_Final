?	???ad@???ad@!???ad@	V??X@V??X@!V??X@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???ad@?m??ʆ??1?Ӝ????IhwH1?@Y	m9????r0*x?&1z}@)      0=2J
Iterator::Root::MapO]?,σ??!<ZC?)P@)@?:s	??1??=fnH@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?`p????!A??@@?0@)?`p????1A??@@?0@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatt&m???!?Fv?.@)f??t牯?1?A??5*@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ȑ??ȫ?!?#'@)?u?X???1?A??(A@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??$?pt??!w???@)??$?pt??1w???@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip???????!?pe?l@@)A?m??1??^?1@:Preprocessing2E
Iterator::RootF??0E??!?GM??P@)???+,??18paZ@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?B?_?+??!?g?1@)?B?_?+??1?g?1@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap??#????!??eV??+@)??֦????1h??6??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?76.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9V??X@I???S@Qf?t?j1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m??ʆ???m??ʆ??!?m??ʆ??      ??!       "	?Ӝ?????Ӝ????!?Ӝ????*      ??!       2      ??!       :	hwH1?@hwH1?@!hwH1?@B      ??!       J		m9????	m9????!	m9????R      ??!       Z		m9????	m9????!	m9????b      ??!       JGPUYV??X@b q???S@yf?t?j1@?"7
sequential/dense_1/MatMulMatMul?#B?e???!?#B?e???0"7
sequential/dense_2/MatMulMatMul?졬????!(?;???0"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentumm?&{֖??!^I?y?1??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum??????!4??f}???"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentumu?c8?2??!?{?t.Ʃ?"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentuma????!??bUή?"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum???^???!?????ޱ?"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum???^???!?hB
V??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentumT?ӊ(???!Ju?S˶?"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentumT?ӊ(???!?e@??Q      Y@Y??}?	@@a"5?x+?P@qEʴemV@y??NDؤ??"?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?76.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?89.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 