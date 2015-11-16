package Users;
import KNN.KNN;

import NaiveBayes.NaiveBayes;
import SelectData.Crossvalidation;
import SelectData.NWData;
import SelectData.Data;
import ShowData.ConfusionMatrix;

import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Sebastian {
    public void run() {
        NaiveBayes Bayes = new NaiveBayes();
        KNN classifier = new KNN();
        NWData daterino = new NWData();
        Data data;
        Data datatest;
        Crossvalidation validation;
        data = daterino.readCSV("/Users/Sebastian/IdeaProjects/MES_Praktikum/selectedDataVec.csv");
        classifier.datalabel = data.getLabel();
        classifier.dataMatrix = data.getPattern();
        datatest = daterino.readCSV("/Users/Sebastian/IdeaProjects/MES_Praktikum/selectedDatapca500010000.csv");
        double[][] testdata = datatest.getPattern();
        double[] testllabel = datatest.getLabel();
        validation = new Crossvalidation();
        ArrayList<double[][]> patternlist = Crossvalidation.crossvalidate(classifier.dataMatrix,classifier.datalabel,10,10);
        ArrayList<double[]> labellist = Crossvalidation.crossvalidatelabel(classifier.dataMatrix,classifier.datalabel,10,10);
        for (int i = 0; i < 8 ; i++) {
            Bayes.addTrainData(patternlist.get(i),labellist.get(i));
        }

        Bayes.train();

        double[] classifiedlabel = Bayes.classifyalldata(patternlist.get(9));

        ConfusionMatrix mat = new ConfusionMatrix();
        mat.computeConfusionMatrix(classifiedlabel,labellist.get(9));

    }

}
