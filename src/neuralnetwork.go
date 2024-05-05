package main

import (
	"errors"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type neuralNetwork struct {
	config    neuralNetworkConfig
	functions []*mat.Dense
}

func initializeNewNeuralNetwork(config neuralNetworkConfig) *neuralNetwork {
	nn := &neuralNetwork{config: config}
	nn.initFunctions()
	return nn
}

func (nn *neuralNetwork) initFunctions() {
	nn.functions = make([]*mat.Dense, 2)

	nn.functions[0] = mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	randomizeMatrix(nn.functions[0])

	nn.functions[1] = mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	randomizeMatrix(nn.functions[1])
}

func randomizeMatrix(m *mat.Dense) {
	r, c := m.Dims()
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.Float64()
	}
	m.SetRawMatrix(mat.NewDense(r, c, data).RawMatrix())
}

func (nn *neuralNetwork) train(x, y *mat.Dense) error {
	for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
		output, err := nn.forwardPass(x)
		if err != nil {
			return err
		}

		r, c := y.Dims()
		errorMatrix := mat.NewDense(r, c, nil)
		errorMatrix.Sub(y, output)

		gradients, err := nn.backwardPass(x, output, errorMatrix)
		if err != nil {
			return err
		}

		nn.updateFunctions(gradients)
	}
	return nil
}

func (nn *neuralNetwork) forwardPass(x *mat.Dense) (*mat.Dense, error) {
	sigmoid := func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}

	numRows, _ := x.Dims()

	hiddenLayerOutput := mat.NewDense(numRows, nn.config.hiddenNeurons, nil)
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.functions[0])

	hiddenLayerOutput.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, hiddenLayerInput)

	output := mat.NewDense(numRows, nn.config.outputNeurons, nil)
	output.Mul(hiddenLayerOutput, nn.functions[1])

	return output, nil
}

func (nn *neuralNetwork) backwardPass(x, output, errorMatrix *mat.Dense) ([]*mat.Dense, error) {
	sigmoidPrime := func(x float64) float64 {
		return x * (1 - x)
	}
	gradients := make([]*mat.Dense, 2)

	finalLayerGradients := new(mat.Dense)
	finalLayerOutput := new(mat.Dense)
	finalLayerOutput.Apply(func(i, j int, v float64) float64 {
		return sigmoidPrime(v)
	}, output)
	finalLayerGradients.MulElem(errorMatrix, finalLayerOutput)
	gradients[1] = finalLayerGradients

	hiddenLayerGradients := new(mat.Dense)
	hiddenLayerOutput := new(mat.Dense)
	hiddenLayerOutput.Apply(func(i, j int, v float64) float64 {
		return sigmoidPrime(v)
	}, output)

	mul := new(mat.Dense)
	mul.Mul(finalLayerGradients, nn.functions[1].T())

	hiddenLayerGradients.MulElem(mul, hiddenLayerOutput)
	gradients[0] = hiddenLayerGradients

	return gradients, nil
}

func (nn *neuralNetwork) updateFunctions(gradients []*mat.Dense) {
	for i, gradient := range gradients {
		nn.functions[i].Apply(func(j, k int, v float64) float64 {
			return v + nn.config.learningRate*gradient.At(j, k)
		}, nn.functions[i])
	}
}

func (nn *neuralNetwork) predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.functions == nil {
		return nil, errors.New("the supplied functions are empty")
	}

	output := new(mat.Dense)
	input := x

	layerOutput := new(mat.Dense)
	layerInput := new(mat.Dense)
	layerInput.Mul(input, nn.functions[0])
	layerOutput.Apply(func(_, _ int, v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	}, layerInput)

	output.Mul(layerOutput, nn.functions[1])

	return output, nil
}
