package Users;
import KNN.KNN;

import SelectData.Crossvalidation;
import SelectData.NWData;
import SelectData.Data;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Sebastian {
    public void run() {
        KNN classifier = new KNN();
        NWData daterino = new NWData();
        Data data;
        Data datatest;
        Crossvalidation validation;
        data = daterino.readCSV("/Users/Sebastian/IdeaProjects/MES_Praktikum/selectedDatapcaVec1000060000.csv");
        classifier.datalabel = data.getLabel();
        classifier.dataMatrix = data.getPattern();
        datatest = daterino.readCSV("/Users/Sebastian/IdeaProjects/MES_Praktikum/selectedDatapcaVec05000.csv");
        double[][] testdata = datatest.getPattern();
        double[] testllabel = datatest.getLabel();
        validation = new Crossvalidation();
        ArrayList<double[][]> patternlist = Crossvalidation.crossvalidate(classifier.dataMatrix,classifier.datalabel,10,10);
        ArrayList<double[]> labellist = Crossvalidation.crossvalidatelabel(classifier.dataMatrix,classifier.datalabel,10,10);

        for (int t = 1; t < 10; t++) {
            double[][] patterni = patternlist.get(t);
            double[] labeli = labellist.get(t);
            classifier.train(patterni,labeli);
        }

        for (int c = 4; c < 10; c++) {
            int[] prediction = classifier.classifyalldata( c, testdata, "Manhatten");
            int mistakes = 0;
            for (int i = 0; i < prediction.length ; i++) {
                if((prediction[i] - (int) testllabel[i])!=0){
                    mistakes += 1;
                }

            }
            System.out.println( "      K="+c + "  Mistakes= "+mistakes);
        }

    }

}
