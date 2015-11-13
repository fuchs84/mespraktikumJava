package KNN;
import SelectData.NWData;
import SelectData.Crossvalidation;
import SelectData.Data;

import java.util.AbstractList;
import java.util.ArrayList;

/**
 * Created by Sebastian on 10.11.2015.
 */


public class KNN {
    private static NWData NWData;
    private static Data data;
    private static Data datatest;
    public static Crossvalidation validation;
    public static Crossvalidation crnvalidation;


    public double[][] trainMatrix;
    public double[] trainlabel;


    public double[][] dataMatrix;
    public double[] datalabel;
    public double[][] testdata;
    public double[] testlabel;


    public KNN() {
        trainMatrix = new double[0][0];
        trainlabel = new double[0];

    }

    //Klassifiziert set aus Trainingsdaten
    public int[] classifyalldata( int KNNS, double[][] testdata,String distancecalculation){
        int[] predictedlabel = new int[testdata.length];
        for (int i = 0; i < predictedlabel.length; i++) {
            predictedlabel[i] = classify(KNNS, testdata[i],distancecalculation);

        }
        return predictedlabel;
    }

    //fuegt weiter trainingsdaten zum set hinzu
    public void train(double[][] pattern, double[] label){
        double [][] temppattern = trainMatrix;
        double [] templabel = trainlabel;
        trainMatrix = new double[trainMatrix.length+pattern.length][pattern[0].length];
        trainlabel = new double[trainlabel.length+label.length];
        for (int i = 0; i < temppattern.length; i++) {
            trainlabel[i] = templabel[i];
            for (int j = 0; j < trainMatrix[0].length; j++) {
                trainMatrix[i][j] = temppattern[i][j];
            }

        }
        for (int i = temppattern.length; i < temppattern.length+pattern.length; i++) {
            trainlabel[i] = label[i-temppattern.length];
            for (int j = 0; j < trainMatrix[0].length; j++) {
                trainMatrix[i][j] = pattern[i-temppattern.length][j];
            }

        }
    }

    //optimiert trainset durch eliminierung von vermutlichen Fehlern
    public void optimizetrainset(){
        int lengthminus = 0;
        for (int i = 0; i < trainMatrix.length; i++) {
            int predicted = classify(7,trainMatrix[i],"Manhatten");
            if(predicted != (int) trainlabel[i]){
                lengthminus +=1;
            }
        }
        System.out.println(lengthminus);
        double[][] temppattern = new double[trainMatrix.length-lengthminus][trainMatrix[0].length];
        double [] templabel = new double[trainlabel.length-lengthminus];
        System.out.println(temppattern.length);
        int runindex = 0;
        for (int j = 0; j < trainMatrix.length; j++) {
            int predicted = classify(7,trainMatrix[j],"Manhatten");
            if(predicted == (int) trainlabel[j]){
                templabel[runindex] = trainlabel[j];
                for (int i = 0; i < temppattern[0].length; i++) {
                    temppattern[runindex][i] = trainMatrix[j][i];
                }
                runindex+=1;
            }
        }

        trainMatrix = new double[temppattern.length][temppattern[0].length];
        trainlabel = new double[templabel.length];
        trainMatrix = temppattern;
        trainlabel = templabel;
    }

    //Klassifiziert einzelnen Vektor von Daten
    public int classify(  int KNNS, double[] testdata,String distancecalculation) {
        double[][] datapattern = trainMatrix;
        double[] label = trainlabel;
        int abzugelemente = 0;
        double[] distance = new double[label.length];
        if(distancecalculation.equals("Euclidean")){
            for (int i = 0; i < label.length; i++){
                double squaresum = 0;
                for (int j = 0; j < testdata.length-abzugelemente; j++) {
                    squaresum += Math.pow((datapattern[i][j] - testdata[j]), 2);
                }
                distance[i] = Math.sqrt(squaresum);
            }
        }else if (distancecalculation.equals("Manhatten")){
                for (int i = 0; i < label.length; i++){
                    double squaresum = 0;
                    for (int j = 0; j < testdata.length-abzugelemente; j++) {
                        squaresum += Math.abs((datapattern[i][j] - testdata[j]));
                    }
                    distance[i] = squaresum;
                }
        }

        double[] distancecheck = distance;
        int[] extrema = new int[KNNS];
        for (int t = 0; t < KNNS; t++) {
            for (int z = 0; z < distancecheck.length; z++) {
                //System.out.println(distance[z]);
                for (int w = z; w < distancecheck.length; w++) {
                    if (distancecheck[z] > distancecheck[w]) {
                        z = w - 1;
                        break;
                    } else if (w == distancecheck.length - 1) {
                        extrema[t] = z;
                        distancecheck[z] = Integer.MAX_VALUE;
                        z = distancecheck.length + 1;
                        break;
                    }

                }

            }


        }
        double sum;
        sum = 0;
        int[] lablearray = new int[extrema.length];
        for (int f = 0; f < extrema.length; f++) {
            lablearray[f] = (int) label[extrema[f]];
        }
        int singlelabel = getPopularElement(lablearray);


        return singlelabel;
    }


    public int getPopularElement(int[] a) {
        int count = 1, tempCount;
        int popular = a[0];
        int temp = 0;
        for (int i = 0; i < (a.length - 1); i++) {
            temp = a[i];
            tempCount = 0;
            for (int j = 1; j < a.length; j++) {
                if (temp == a[j])
                    tempCount++;
            }
            if (tempCount > count) {
                popular = temp;
                count = tempCount;
            }
        }
        return popular;
    }
}