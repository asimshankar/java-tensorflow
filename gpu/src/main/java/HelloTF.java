import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.framework.ConfigProto;

public class HelloTF {
  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("ERROR: Usage: HelloTF <path_to_GraphDef>");
      System.exit(1);
    }
    try (Graph graph = new Graph()) {
      graph.importGraphDef(Files.readAllBytes(Paths.get(args[0])), "");
      // Create a config that will dump out device placement of operations.
      ConfigProto config = ConfigProto.newBuilder().setLogDevicePlacement(true).build();
      try (Session sess = new Session(graph, config.toByteArray())) {
        try (Tensor in = Tensor.create(new float[][] {{1, 2}, {3, 4}});
            Tensor out = sess.runner().feed("input", in).fetch("output").run().get(0)) {
          System.out.println("TensorFlow version: " + TensorFlow.version());
          System.out.println();
          print2x2Matrix("Input ", in);
          print2x2Matrix("Output", out);
        }
      }
    }
  }

  public static void print2x2Matrix(String tag, Tensor t) {
    float[][] m = new float[2][2];
    System.out.print(tag + ": ");
    System.out.println(Arrays.deepToString(t.copyTo(m)));
  }
}
