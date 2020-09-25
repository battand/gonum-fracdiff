package fracdiff

import (
	"testing"

	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

func TestFracDiffSuite(t *testing.T) {
	suite.Run(t, new(FracDiffTestSuite))
}

type FracDiffTestSuite struct {
	suite.Suite
	fracDiffer FracDiffer
}

func (s *FracDiffTestSuite) SetupTest() {
	s.fracDiffer = &FracDiff{}
}

func (s *FracDiffTestSuite) TestWeightsHappy() {
	weights, err := s.fracDiffer.GetWeights(.75, 5, 0)
	s.Nil(err, "err is nil")
	expected := mat.NewDense(1, 5, []float64{
		-0.02197265625,
		-0.0390625,
		-0.09375,
		-0.75,
		1,
	})
	s.Equal(weights, expected)
}

func (s *FracDiffTestSuite) TestWeightsThreshold() {
	weights, err := s.fracDiffer.GetWeights(.75, 5, 0.05)
	s.Nil(err, "err is nil")
	expected := mat.NewDense(1, 3, []float64{
		-0.09375,
		-0.75,
		1,
	})
	s.Equal(weights, expected)
}

func (s *FracDiffTestSuite) TestGetWeightsDError() {
	weights, err := s.fracDiffer.GetWeights(-.01, 20, 0.5)
	s.Nil(weights, "weights is nil")
	s.EqualError(err, "d must be between 0 and 1 inclusive", "error matches")
	weights, err = s.fracDiffer.GetWeights(1.01, 20, 0.5)
	s.Nil(weights, "weights is nil")
	s.EqualError(err, "d must be between 0 and 1 inclusive", "error matches")
}

func (s *FracDiffTestSuite) TestGetWeightsWindowSizeError() {
	weights, err := s.fracDiffer.GetWeights(.75, 0, 0.5)
	s.Nil(weights, "weights is nil")
	s.EqualError(err, "windowSize must be >= 1", "error matches")
}

func testSeries() *mat.Dense {
	vals := []float64{}
	for i := 100; i < 200; i++ {
		vals = append(vals, float64(i))
	}
	return mat.NewDense(len(vals), 1, vals)
}

var expectedDifferentiatedSeries = []float64{0, 0, 0, 0, 11.044921875, 11.14013671875, 11.2353515625, 11.33056640625, 11.42578125, 11.52099609375, 11.6162109375, 11.71142578125, 11.806640625, 11.90185546875, 11.9970703125, 12.09228515625, 12.1875, 12.28271484375, 12.3779296875, 12.47314453125, 12.568359375, 12.66357421875, 12.7587890625, 12.85400390625, 12.94921875, 13.04443359375, 13.1396484375, 13.23486328125, 13.330078125, 13.42529296875, 13.5205078125, 13.61572265625, 13.7109375, 13.80615234375, 13.9013671875, 13.99658203125, 14.091796875, 14.18701171875, 14.2822265625, 14.37744140625, 14.47265625, 14.56787109375, 14.6630859375, 14.75830078125, 14.853515625, 14.94873046875, 15.0439453125, 15.13916015625, 15.234375, 15.32958984375, 15.4248046875, 15.52001953125, 15.615234375, 15.71044921875, 15.8056640625, 15.90087890625, 15.99609375, 16.09130859375, 16.1865234375, 16.28173828125, 16.376953125, 16.47216796875, 16.5673828125, 16.66259765625, 16.7578125, 16.85302734375, 16.9482421875, 17.04345703125, 17.138671875, 17.23388671875, 17.3291015625, 17.42431640625, 17.51953125, 17.61474609375, 17.7099609375, 17.80517578125, 17.900390625, 17.99560546875, 18.0908203125, 18.18603515625, 18.28125, 18.37646484375, 18.4716796875, 18.56689453125, 18.662109375, 18.75732421875, 18.8525390625, 18.94775390625, 19.04296875, 19.13818359375, 19.2333984375, 19.32861328125, 19.423828125, 19.51904296875, 19.6142578125, 19.70947265625, 19.8046875, 19.89990234375, 19.9951171875, 20.09033203125}

func (s *FracDiffTestSuite) TestDifferentiateHappy() {
	series := testSeries()
	result, err := s.fracDiffer.Differentiate(.75, 5, 0, series)
	s.Nil(err, "err is nil")

	expected := mat.NewDense(len(expectedDifferentiatedSeries), 1, expectedDifferentiatedSeries)
	s.Equal(result, expected)
}

func (s *FracDiffTestSuite) TestDifferentiateError() {
	series := testSeries()
	result, err := s.fracDiffer.Differentiate(-1, 20, 0, series)
	s.Nil(result, "expected nil result")
	s.EqualError(err, "d must be between 0 and 1 inclusive", "error equals")
}
