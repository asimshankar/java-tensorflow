import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class Train {

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("Require two arguments: <graph_def_filename> <directory_for_checkpoints>");
      System.exit(1);
    }
    final byte[] graphDef = Files.readAllBytes(Paths.get(args[0]));
    final String checkpointDir = args[1];
    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

    // These names of tensors/operations in the graph (string arguments to feed(), fetch(), and
    // addTarget()) would have been printed out by model.py
    try (Graph graph = new Graph();
        Session sess = new Session(graph);
        Tensor<String> checkpointPrefix =
            Tensors.create(Paths.get(checkpointDir, "checkpoint").toString())) {
      graph.importGraphDef(graphDef);

      // Initialize or restore.
      if (checkpointExists) {
        System.out.println("Restoring variables from checkpoint");
        sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
      } else {
        System.out.println("Initializing variables");
        sess.runner().addTarget("init").run();
      }

      System.out.println("Generating initial predictions");
      printPredictionsOnTestSet(sess);

      System.out.println("Training for a few steps");
      final int BATCH_SIZE = 10;
      float inputs[][][] = new float[BATCH_SIZE][1][1];
      float targets[][][] = new float[BATCH_SIZE][1][1];
      for (int i = 0; i < 200; ++i) {
        fillNextBatchForTraining(inputs, targets);
        try (Tensor<Float> inputBatch = Tensors.create(inputs);
            Tensor<Float> targetBatch = Tensors.create(targets)) {
          sess.runner()
              .feed("input", inputBatch)
              .feed("target", targetBatch)
              .addTarget("train")
              .run();
        }
      }

      System.out.println("Updated predictions");
      printPredictionsOnTestSet(sess);

      System.out.println("Saving checkpoint");
      sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
    }
  }

  public static void printPredictionsOnTestSet(Session sess) {
    final float[][][] inputBatch = new float[][][] {{{1.0f}}, {{2.0f}}, {{3.0f}}};
    try (Tensor<Float> input = Tensors.create(inputBatch);
        Tensor<Float> output =
            sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
      final long shape[] = output.shape();
      final int batchSize = (int) shape[0];
      final int rows = (int) shape[1];
      final int cols = (int) shape[2];
      float[][][] predictions = output.copyTo(new float[batchSize][rows][cols]);
      for (int i = 0; i < batchSize; ++i) {
        System.out.print("\t x = ");
        System.out.print(Arrays.deepToString(inputBatch[i]));
        System.out.print(", predicted y = ");
        System.out.println(Arrays.deepToString(predictions[i]));
      }
    }
  }

  public static void fillNextBatchForTraining(float[][][] inputs, float[][][] targets) {
    final Random r = new Random();
    for (int i = 0; i < inputs.length; ++i) {
      inputs[i][0][0] = r.nextFloat();
      targets[i][0][0] = inputs[i][0][0] * 3.0f + 2.0f;
    }
  }
}
