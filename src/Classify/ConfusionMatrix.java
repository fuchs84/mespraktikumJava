package Classify;

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
    private int[][] matrixNB = new int[3][3];
    private int[][] matrixMLP = new int[3][3];
    private int trueNB = 0;
    private int falseNB = 0;
    private int trueMLP = 0;
    private int falseMLP = 0;

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

    public void resultsNB(double[] output, double[] desiredOutput){
        int tempDesired;
        int tempOutput;
        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrixNB[tempDesired-1][tempOutput-1]++;
        }
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueNB++;
            } else {
                falseNB++;
            }
        }
    }

    public void resultsMLP(double[] output, double[] desiredOutput){
        int tempDesired;
        int tempOutput;
        for (int h = 0; h < desiredOutput.length; h++){
            tempDesired = (int) desiredOutput[h];
            tempOutput = (int) output[h];
            matrixMLP[tempDesired-1][tempOutput-1]++;
        }
        for (int i = 0; i < output.length; i++) {
            if(output[i] == desiredOutput[i]) {
                trueMLP++;
            } else {
                falseMLP++;
            }
        }
    }


    public void printEntireResults(StringBuilder stringBuilder) {
        String temp;
        stringBuilder.append("Confusion-Matrix KNN: " + "\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp = String.format("%6d ", matrixKNN[i][j]);
                stringBuilder.append(temp);

            }
            stringBuilder.append("\n");
        }
        stringBuilder.append("Results KNN: " + "\n");
        stringBuilder.append("True: " + trueKNN + "\n");
        stringBuilder.append("False: " + falseKNN + "\n");
        stringBuilder.append("Percent: " + ((double) trueKNN / (trueKNN + falseKNN)) + "\n\n\n");

        stringBuilder.append("Confusion-Matrix Binary-DT: " + "\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp = String.format("%6d ", matrixBinary[i][j]);
                stringBuilder.append(temp);
            }
            System.out.println();
        }
        stringBuilder.append("Results Binary-DT: " + "\n");
        stringBuilder.append("True: " + trueBinary + "\n");
        stringBuilder.append("False: " + falseBinary + "\n");
        stringBuilder.append("Percent: " + ((double) trueBinary / (trueBinary + falseBinary)) + "\n\n\n");

        stringBuilder.append("Confusion-Matrix Multi-DT: " + "\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp = String.format("%6d ", matrixMulti[i][j]);
                stringBuilder.append(temp);
            }
            stringBuilder.append("\n");
        }
        stringBuilder.append("Results Multi-DT: " + "\n");
        stringBuilder.append("True: " + trueMulti + "\n");
        stringBuilder.append("False: " + falseMulti + "\n");
        stringBuilder.append("Percent: " + ((double) trueMulti / (trueMulti + falseMulti)) + "\n\n\n");

        stringBuilder.append("Confusion-Matrix NaiveBayes: " + "\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp = String.format("%6d ", matrixNB[i][j]);
                stringBuilder.append(temp);
            }
            stringBuilder.append("\n");
        }
        stringBuilder.append("Results NaiveBayes: " + "\n");
        stringBuilder.append("True: " + trueNB + "\n");
        stringBuilder.append("False: " + falseNB + "\n");
        stringBuilder.append("Percent: " + ((double) trueNB / (trueNB + falseNB)) + "\n\n\n");

        stringBuilder.append("Confusion-Matrix MLP: " + "\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                temp = String.format("%6d ", matrixMLP[i][j]);
                stringBuilder.append(temp);
            }
            stringBuilder.append("\n");
        }
        stringBuilder.append("Results MLP: " + "\n");
        stringBuilder.append("True: " + trueMLP + "\n");
        stringBuilder.append("False: " + falseMLP + "\n");
        stringBuilder.append("Percent: " + ((double) trueMLP / (trueMLP + falseMLP)) + "\n\n\n");
    }
}
