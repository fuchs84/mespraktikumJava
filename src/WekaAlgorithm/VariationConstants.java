package WekaAlgorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;

/**
 * Variation the variables for the best result
 */
public class VariationConstants {
    Controller controller;

    /**
     * Method varies for the classifiers and mistakes the variables of each classifier
     * @throws Exception to invoking Method
     */
    public void automaticVariation() throws Exception {
        String patternPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMix.csv";
        String labelPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMix.csv";
        String patternPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMixpass.csv";
        String labelPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMixpass.csv";

        String[][] classifiers =  {{"AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"
                , "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"},
                {"DecisionTree", "DecisionTree","DecisionTree","DecisionTree","DecisionTree","DecisionTree","DecisionTree",
                        "DecisionTree","DecisionTree","DecisionTree","DecisionTree"},
                {"RandomForest","RandomForest","RandomForest","RandomForest", "RandomForest","RandomForest", "RandomForest",
                        "RandomForest","RandomForest","RandomForest","RandomForest",},
                {"NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes",
                        "NaiveBayes","NaiveBayes","NaiveBayes"},
                {"KNN","KNN","KNN","KNN","KNN","KNN","KNN","KNN",
                        "KNN","KNN","KNN"}};

        FileWriter fw = new FileWriter("variation.txt");



        controller = new Controller();
        controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
        for(int i = 2; i < 6; i++) {
            for(int j = 1; j < 6; j++) {
                String[] option = {"-C", Integer.toString(i), "-I", Integer.toString(j*5)};
                fw.append("AdaBoost: \t");
                for(int k = 0; k < option.length; k++) {
                    fw.append(option[k] + " ");
                }
                fw.append("\t");
                String[][] options = extendOptions(option);
                controller.setSelectedClassifiers(classifiers[0]);
                controller.setSelectedOptions(options);
                controller.buildStructure();
                controller.evaluation("crossValidation");
                double[] results = controller.getEvaluationResults();
                for(int k = 0; k < results.length; k++) {
                    fw.append(Double.toString(results[k]) + "\t");
                }
                fw.append("\n");
            }
        }
        fw.append("\n");

        for(int i = 0; i < 2; i++) {
            if(i == 0) {
                String[] option = new String[1];
                option[0] = "-U";
                fw.append("Decision Tree: \t");
                for(int k = 0; k < option.length; k++) {
                    fw.append(option[k] + " ");
                }
                fw.append("\t");
                String[][] options = extendOptions(option);
                controller.setSelectedClassifiers(classifiers[1]);
                controller.setSelectedOptions(options);
                controller.buildStructure();
                controller.evaluation("crossValidation");
                double[] results = controller.getEvaluationResults();
                for(int k = 0; k < results.length; k++) {
                    fw.append(Double.toString(results[k]) + "\t");
                }
                fw.append("\n");
            } else {
                String[] option = new String[2];
                option[0] = "-I";
                for(int j = 1; j < 11; j++) {
                    option[1] = Double.toString(j*0.05);
                    fw.append("Decision Tree: \t");
                    for(int k = 0; k < option.length; k++) {
                        fw.append(option[k] + " ");
                    }
                    fw.append("\t");
                    String[][] options = extendOptions(option);
                    controller.setSelectedClassifiers(classifiers[1]);
                    controller.setSelectedOptions(options);
                    controller.buildStructure();
                    controller.evaluation("crossValidation");
                    double[] results = controller.getEvaluationResults();
                    for(int k = 0; k < results.length; k++) {
                        fw.append(Double.toString(results[k]) + "\t");
                    }
                    fw.append("\n");
                }
            }
        }
        fw.append("\n");

        for(int i = 1; i < 11; i++) {
            String[] option = {"-I", Integer.toString(i*10)};
            fw.append("Random Forest: \t");
            for(int k = 0; k < option.length; k++) {
                fw.append(option[k] + " ");
            }
            fw.append("\t");
            String[][] options = extendOptions(option);
            controller.setSelectedClassifiers(classifiers[2]);
            controller.setSelectedOptions(options);
            controller.buildStructure();
            controller.evaluation("crossValidation");
            double[] results = controller.getEvaluationResults();
            for(int k = 0; k < results.length; k++) {
                fw.append(Double.toString(results[k]) + "\t");
            }
            fw.append("\n");
        }
        fw.append("\n");

        for(int i = 0; i < 2; i++) {
            String[] option = new String[1];
            if(i == 0) {
                option[0] = "-K";
            } else {
                option[0] = "-D";
            }
            fw.append("Naive Bayes: \t");
            for(int k = 0; k < option.length; k++) {
                fw.append(option[k] + " ");
            }
            fw.append("\t");
            String[][] options = extendOptions(option);
            controller.setSelectedClassifiers(classifiers[3]);
            controller.setSelectedOptions(options);
            controller.buildStructure();
            controller.evaluation("crossValidation");
            double[] results = controller.getEvaluationResults();
            for(int k = 0; k < results.length; k++) {
                fw.append(Double.toString(results[k]) + "\t");
            }
            fw.append("\n");
        }
        fw.append("\n");


        for(int i = 1; i < 11; i++) {
            String[] option = {"-K", Integer.toString(i)};
            fw.append("KNN:  \t");
            for(int k = 0; k < option.length; k++) {
                fw.append(option[k] + " ");
            }
            fw.append("\t");
            String[][] options = extendOptions(option);
            controller.setSelectedClassifiers(classifiers[4]);
            controller.setSelectedOptions(options);
            controller.buildStructure();
            controller.evaluation("crossValidation");
            double[] results = controller.getEvaluationResults();
            for(int k = 0; k < results.length; k++) {
                fw.append(Double.toString(results[k]) + "\t");
            }
            fw.append("\n");
        }
        fw.close();
    }

    /**
     * Method extends the Options for the automatic variation
     * @param option 1d-array with the options
     * @return 2d-array with the options
     */
    private String[][] extendOptions(String[] option) {
        String[][] options  = new String[11][];
        for(int i = 0; i < options.length; i++) {
            options[i] = option;
        }
        return options;
    }
}
