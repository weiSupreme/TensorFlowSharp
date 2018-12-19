using System.IO;
using TensorFlow;

namespace ExampleCommon
{
	public static class ImageUtil
	{
        static byte[] contents = new byte[656*490];
        static byte[] crop = new byte[336 * 336];
        // Convert the image in filename to a Tensor suitable as input to the Inception model.
        public static TFTensor CreateTensorFromImageFile (string file, TFDataType destinationDataType = TFDataType.Float)
		{
			contents = File.ReadAllBytes (file);
            System.Buffer.BlockCopy(contents, 99712, crop, 0, 336 * 336);

			// DecodeJpeg uses a scalar String-valued tensor as input.
			var tensor = TFTensor.CreateString (contents);

			TFOutput input, output;

			// Construct a graph to normalize the image
			using (var graph = ConstructGraphToNormalizeImage (out input, out output, destinationDataType)){
				// Execute that graph to normalize this one image
				using (var session = new TFSession (graph)) {
					var normalized = session.Run (
						inputs: new [] { input },
						inputValues: new [] { tensor },
						outputs: new [] { output });
					
					return normalized [0];
				}
			}
		}

		// The inception model takes as input the image described by a Tensor in a very
		// specific normalized format (a particular image size, shape of the input tensor,
		// normalized pixel values etc.).
		//
		// This function constructs a graph of TensorFlow operations which takes as
		// input a JPEG-encoded string and returns a tensor suitable as input to the
		// inception model.
		private static TFGraph ConstructGraphToNormalizeImage (out TFOutput input, out TFOutput output, TFDataType destinationDataType = TFDataType.Float)
		{
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained after with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			//   float using (value - Mean)/Scale.

			const int W = 224;
			const int H = 224;
            const float Mean = 0; // 117;
			const float Scale = 255f;

			var graph = new TFGraph ();
			input = graph.Placeholder (TFDataType.String);

			output = graph.Cast (graph.Div (
				x: graph.Sub (
					x: graph.ResizeBilinear (
						images: graph.ExpandDims (
							input: graph.Cast (
								graph.DecodeBmp (contents: input, channels: 1), DstT: TFDataType.Float),
							dim: graph.Const (0, "make_batch")),
						size: graph.Const (new int [] { W, H }, "size")),
					y: graph.Const (Mean, "mean")),
				y: graph.Const (Scale, "scale")), destinationDataType);
			
			return graph;
		}
	}
}
