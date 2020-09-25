package fracdiff

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

type FracDiffer interface {
	GetWeights(d float64, windowSize int, threshold float64) (*mat.Dense, error)
	Differentiate(d float64, windowSize int, threshold float64, data *mat.Dense) (*mat.Dense, error)
}

type FracDiff struct {
}

func (f *FracDiff) GetWeights(d float64, windowSize int, threshold float64) (*mat.Dense, error) {
	if d < 0 || d > 1 {
		return nil, errors.New("d must be between 0 and 1 inclusive")
	}
	if windowSize < 1 {
		return nil, errors.New("windowSize must be >= 1")
	}
	weights := []float64{}
	for index := 0; index < windowSize; index++ {
		if index == 0 {
			weights = append(weights, 1)
		} else {
			weight := weights[index-1] * -1 * ((d - float64(index) + 1) / float64(index))
			if math.Abs(weight) < threshold {
				break
			}
			weights = append(weights, weight)
		}
	}
	weightsLen := len(weights)
	reversed := make([]float64, weightsLen)
	for index := range weights {
		reversed[index] = weights[weightsLen-index-1]
	}
	return mat.NewDense(1, weightsLen, reversed), nil
}

func (f *FracDiff) Differentiate(d float64, windowSize int, threshold float64, data *mat.Dense) (*mat.Dense, error) {
	weights, err := f.GetWeights(d, windowSize, threshold)
	if err != nil {
		return nil, err
	}
	_, weightsLen := weights.Dims()
	dataLen, _ := data.Dims()
	output := mat.NewDense(dataLen, 1, nil)

	for i := 0; i <= (dataLen - weightsLen); i++ {
		window := data.Slice(i, weightsLen+i, 0, 1)

		var result mat.Dense
		result.Mul(weights, window)
		differentiated := result.At(0, 0)
		output.Set(i+weightsLen-1, 0, differentiated)
	}

	return output, nil
}
