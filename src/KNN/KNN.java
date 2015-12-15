package KNN;

/**
 * Created by Sebastian on 10.11.2015.
 */


public class KNN {



    private double[][] trainPatterns;
    private double[] trainLabels;


    public KNN() {
        trainPatterns = new double[0][0];
        trainLabels = new double[0];

    }

    //Klassifiziert set aus Trainingsdaten
    public double[] classify( int knn, double[][] patterns,String distanceCalculation){
        double [] predictedlabel = new double[patterns.length];
        for (int i = 0; i < predictedlabel.length; i++) {
            predictedlabel[i] = classify(knn, patterns[i],distanceCalculation);

        }
        return predictedlabel;
    }

    //fuegt weiter trainingsdaten zum set hinzu
    public void train(double[][] patterns, double[] labels){
        double [][] temppattern = trainPatterns;
        double [] templabel = trainLabels;
        trainPatterns = new double[trainPatterns.length+patterns.length][patterns[0].length];
        trainLabels = new double[trainLabels.length+labels.length];
        for (int i = 0; i < temppattern.length; i++) {
            trainLabels[i] = templabel[i];
            for (int j = 0; j < trainPatterns[0].length; j++) {
                trainPatterns[i][j] = temppattern[i][j];
            }

        }
        for (int i = temppattern.length; i < temppattern.length+patterns.length; i++) {
            trainLabels[i] = labels[i-temppattern.length];
            for (int j = 0; j < trainPatterns[0].length; j++) {
                trainPatterns[i][j] = patterns[i-temppattern.length][j];
            }

        }
    }

    //optimiert trainset durch eliminierung von vermutlichen Fehlern
    public void optimizeTrainSet(){
        int lengthminus = 0;
        for (int i = 0; i < trainPatterns.length; i++) {
            int predicted = classify(7, trainPatterns[i],"Manhattan");
            if(predicted != (int) trainLabels[i]){
                lengthminus +=1;
            }
        }
        System.out.println(lengthminus);
        double[][] temppattern = new double[trainPatterns.length-lengthminus][trainPatterns[0].length];
        double [] templabel = new double[trainLabels.length-lengthminus];
        System.out.println(temppattern.length);
        int runindex = 0;
        for (int j = 0; j < trainPatterns.length; j++) {
            int predicted = classify(7, trainPatterns[j],"Manhattan");
            if(predicted == (int) trainLabels[j]){
                templabel[runindex] = trainLabels[j];
                for (int i = 0; i < temppattern[0].length; i++) {
                    temppattern[runindex][i] = trainPatterns[j][i];
                }
                runindex+=1;
            }
        }

        trainPatterns = new double[temppattern.length][temppattern[0].length];
        trainLabels = new double[templabel.length];
        trainPatterns = temppattern;
        trainLabels = templabel;
    }

    //Klassifiziert einzelnen Vektor von Daten
    private int classify(  int knn, double[] pattern,String distanceCalculation) {
        double[][] datapattern = trainPatterns;
        double[] label = trainLabels;
        int abzugelemente = 0;
        double[] distance = new double[label.length];
        if(distanceCalculation.equals("Euclidean")){
            for (int i = 0; i < label.length; i++){
                double squaresum = 0;
                for (int j = 0; j < pattern.length-abzugelemente; j++) {
                    squaresum += Math.pow((datapattern[i][j] - pattern[j]), 2);
                }
                distance[i] = Math.sqrt(squaresum);
            }
        }else if (distanceCalculation.equals("Manhattan")){
                for (int i = 0; i < label.length; i++){
                    double squaresum = 0;
                    for (int j = 0; j < pattern.length-abzugelemente; j++) {
                        squaresum += Math.abs((datapattern[i][j] - pattern[j]));
                    }
                    distance[i] = squaresum;
                }
        }

        double[] distancecheck = distance;
        int[] extrema = new int[knn];
        for (int t = 0; t < knn; t++) {
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
        int[] lablearray = new int[extrema.length];
        for (int f = 0; f < extrema.length; f++) {
            lablearray[f] = (int) label[extrema[f]];
        }
        int singlelabel = getPopularElement(lablearray);


        return singlelabel;
    }


    private int getPopularElement(int[] a) {
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