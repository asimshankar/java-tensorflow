# Vector of strings

As of release 1.3.0 (August 2017), the TensorFlow Java API didn't have an method
to create a `Tensor` of strings for any shapes other than scalars (see
[#8531](\(https://github.com/tensorflow/tensorflow/issues/8531\))).

That issue will hopefully be resolved by 1.4 or 1.5, making it simpler to create
non-scalar string tensors. However, in the mean time this demonstrates a
workaround - use the Java API to construct a TensorFlow model to convert a
scalar string into a vector of one string.

The resulting vector can then be fed into other models that expect a
vector-of-string. For example, models that use the `Example` protocol buffer,
typically using
[`tf.parse_example`](https://www.tensorflow.org/api_docs/python/tf/parse_example)
when building the model in Python.
([e.g.](https://stackoverflow.com/questions/45746742/how-can-i-create-a-tensor-from-an-example-object-in-java)).

See [source](src/main/java/VectorString.java), and run using:

```
mvn -q compile exec:java
```
