package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

func main() {
	fmt.Println("KANGo init successful!")

	f, err := os.Open("../data/training_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputsLoaded := false
	labelsLoaded := false

	if len(inputsData) > 1 {
		inputsLoaded = true
	}

	if len(labelsData) > 1 {
		labelsLoaded = true
	}

	fmt.Printf("Inputs Loaded?: %v\n", inputsLoaded)
	fmt.Printf("Labels Loaded?: %v\n", labelsLoaded)

}
