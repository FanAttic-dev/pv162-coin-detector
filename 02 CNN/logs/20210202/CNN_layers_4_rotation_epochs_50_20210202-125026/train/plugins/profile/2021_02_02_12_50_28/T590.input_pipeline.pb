	�3�c�M�@�3�c�M�@!�3�c�M�@	��(Spq?��(Spq?!��(Spq?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�3�c�M�@�s}�?A��w�GM�@Y�'���?*	��"���2@2F
Iterator::Model�� �rh�?!^�Ŗ��V@)`s�	M�?1�����G@:Preprocessing2P
Iterator::Model::Prefetch�3�ۃ�?!<��3)�E@)�3�ۃ�?1<��3)�E@:Preprocessing2Y
"Iterator::Model::Prefetch::ShuffleL�'��Z?!�Ii!@)L�'��Z?1�Ii!@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��(Spq?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�s}�?�s}�?!�s}�?      ��!       "      ��!       *      ��!       2	��w�GM�@��w�GM�@!��w�GM�@:      ��!       B      ��!       J	�'���?�'���?!�'���?R      ��!       Z	�'���?�'���?!�'���?JCPU_ONLYY��(Spq?b 