package SelectData;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 13.12.15.
 */
public class ReadData {

    public Data readCSVs(String patternPath, String labelPath) {
        double[][] pattern;
        double[][] label;
        ArrayList<String> patternData = new ArrayList<>();
        ArrayList<String> labelData = new ArrayList<>();
        int lineIndex = 0;
        try {
            BufferedReader brP = new BufferedReader(new FileReader(patternPath));
            BufferedReader brL = new BufferedReader(new FileReader(labelPath));


            String patternLine;
            String labelLine;

            while ((patternLine = brP.readLine()) != null && (labelLine = brL.readLine()) != null) {
                patternData.add(patternLine);
                labelData.add(labelLine);
                lineIndex++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int offset = 0;
        int offsetLabels = 4;
        int offsetPatterns = 4;
        String[] labelParts = labelData.get(0).split(",");
        String[] patternParts = patternData.get(0).split(",");
        pattern = new double[lineIndex - offset][patternParts.length-offsetPatterns];
        label = new double[lineIndex - offset][labelParts.length-offsetLabels];

        for(int i = offset; i < lineIndex; i++) {
            patternParts = patternData.get(i).split(",");
            labelParts = labelData.get(i).split(",");
            if(pattern[0].length > label[0].length) {
                for(int j = 0; j < pattern[0].length; j++) {
                    pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetPatterns]);
                    if(j < label[0].length) {
                        label[i-offset][j] = Double.parseDouble(labelParts[j+offsetLabels]);
                    }
                }
            } else {
                for(int j = 0; j < label[0].length; j++) {
                    label[i-offset][j] = Double.parseDouble(labelParts[j+offsetLabels]);
                    if(j < pattern[0].length) {
                        pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetPatterns]);
                    }
                }
            }
        }
        return new Data(label, pattern);
    }


    public int[] readStepCSV(String path) {
        int[] steps;
        ArrayList<String> stepData = new ArrayList<>();
        int lineIndex = 0;
        try {
            BufferedReader brP = new BufferedReader(new FileReader(path));
            String line;
            while ((line = brP.readLine()) != null) {
                stepData.add(line);
                lineIndex++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        String[] lineParts;

        steps = new int[lineIndex];
        int offset = 0;
        for(int i = 0; i < lineIndex; i++) {
            lineParts = stepData.get(i).split("\t");
            steps[i] = (int)(Double.parseDouble(lineParts[offset + 1]) - Double.parseDouble(lineParts[offset]));
        }
        return  steps;
    }
}
