package ShowData;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class ConfusionMatrix {
    private int[][] matrix;
    private int[][] matrixBinary = new int[3][3];
    private int[][] matrixKNN = new int[3][3];
    private int[][] matrixMulti = new int[3][3];
    private int trueBinary = 0;
    private int falseBinary = 0;
    private int trueKNN = 0;
    private int falseKNN = 0;
    private int trueMulti = 0;
    private int falseMulti = 0;

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

    public void resultsKNN(double[] output, double[] desiredOutput){
        int tempDesired;
        int tempOutput;
        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrixKNN[tempDesired-1][tempOutput-1]++;
        }
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueKNN++;
            } else {
                falseKNN++;
            }
        }
    }

    public void resultsBinary(double[] output, double[] desiredOutput){
        int tempDesired;
        int tempOutput;
        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrixBinary[tempDesired-1][tempOutput-1]++;
        }
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueBinary++;
            } else {
                falseBinary++;
            }
        }
    }

    public void resultsMulti(double[] output, double[] desiredOutput){
        int tempDesired;
        int tempOutput;
        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrixMulti[tempDesired-1][tempOutput-1]++;
        }
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueMulti++;
            } else {
                falseMulti++;
            }
        }
    }

    public void printEntireResults() {
        System.out.println("Confusion-Matrix KNN: ");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.printf("%6d ", matrixKNN[i][j]);
            }
            System.out.println();
        }
        System.out.println("Results KNN: ");
        System.out.println("True: " + trueKNN);
        System.out.println("False: " + falseKNN);
        System.out.printf("Percent %.2f", ((double) trueKNN / (trueKNN + falseKNN)));
        System.out.println();
        System.out.println();

        System.out.println("Confusion-Matrix Binary: ");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.printf("%6d ", matrixBinary[i][j]);
            }
            System.out.println();
        }
        System.out.println("Results Binary: ");
        System.out.println("True: " + trueBinary);
        System.out.println("False: " + falseBinary);
        System.out.printf("Percent %.2f", ((double) trueBinary / (trueBinary + falseBinary)));
        System.out.println();
        System.out.println();

        System.out.println("Confusion-Matrix Multi: ");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.printf("%6d ", matrixMulti[i][j]);
            }
            System.out.println();
        }
        System.out.println("Results Multi: ");
        System.out.println("True: " + trueMulti);
        System.out.println("False: " + falseMulti);
        System.out.printf("Percent %.2f", ((double) trueMulti / (trueMulti + falseMulti)));
        System.out.println();
        System.out.println();

    }
}
