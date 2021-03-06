package Classify;

/**
 * Class calculates the confusion matrix
 */
public class ConfusionMatrix {

    //Confusion matrix
    private int[][] matrix;

    /**
     * Method calculates the confusion matrix of a given classified label-set and desired label-set.
     * @param output Classified label-set
     * @param desiredOutput Desired label-set
     */
    public void computeConfusionMatrix(double[] output, double[] desiredOutput) {
        if (output.length != desiredOutput.length) {
            System.out.println("Output und desiredOutput passen nicht zusammen!");
            return;
        }
        System.out.println("Confusionsmatrix: ");
        int max = 0;
        for (int i = 0; i < output.length; i++) {
            if (max < (int) output[i]) {
                max = (int) output[i];
            }
        }

        for (int i = 0; i < desiredOutput.length; i++) {
            if (max < (int) desiredOutput[i]) {
                max = (int) desiredOutput[i];
            }
        }

        matrix = new int[max][max];
        int tempDesired;
        int tempOutput;

        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrix[tempDesired-1][tempOutput-1]++;
        }

        for (int i = 0; i < max; i++) {
            for (int j = 0; j < max; j++) {
                System.out.printf("%6d ", matrix[i][j]);
            }
            System.out.println();
        }
    }

    /**
     * Method calculates the percentage true and false classified Value.
     * @param output Classified label-set
     * @param desiredOutput Desired label-set
     */
    public void computeTrueFalse(double[] output, double[] desiredOutput) {
        if (output.length != desiredOutput.length) {
            System.out.println("Output und desiredOutput passen nicht zusammen!");
            return;
        }
        int trueClassified = 0, falseClassified = 0;
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueClassified++;
            } else {
                falseClassified++;
            }
        }
        System.out.println("Results: ");
        System.out.println("True: " + trueClassified);
        System.out.println("False: " + falseClassified);
        System.out.printf("Percent %.2f", ((double) trueClassified / output.length));
        System.out.println();
    }
}
