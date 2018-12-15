// An example for using the TensorFlow C# API for image recognition
// using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
// 
// Sample usage: <program> -dir=/tmp/modeldir imagefile
// 
// The pre-trained model takes input in the form of a 4-dimensional
// tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
// where:
// - BATCH_SIZE allows for inference of multiple images in one pass through the graph
// - IMAGE_HEIGHT is the height of the images on which the model was trained
// - IMAGE_WIDTH is the width of the images on which the model was trained
// - 3 is the (R, G, B) values of the pixel colors represented as a float.
// 
// And produces as output a vector with shape [ NUM_LABELS ].
// output[i] is the probability that the input image was recognized as
// having the i-th label.
// 
// A separate file contains a list of string labels corresponding to the
// integer indices of the output.
// 
// This example:
// - Loads the serialized representation of the pre-trained model into a Graph
// - Creates a Session to execute operations on the Graph
// - Converts an image file to a Tensor to provide as input to a Session run
// - Executes the Session and prints out the label with the highest probability
// 
// To convert an image file to a Tensor suitable for input to the Inception model,
// this example:
// - Constructs another TensorFlow graph to normalize the image into a
//   form suitable for the model (for example, resizing the image)
// - Creates an executes a Session to obtain a Tensor in this normalized form.
using System;
using TensorFlow;
using Mono.Options;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Collections.Generic;
using ExampleCommon;
using System.Collections.Generic;

namespace ExampleInceptionInference
{
	class MainClass
	{
		public static void Main (string [] args)
		{			
            TFSessionOptions options = new TFSessionOptions();
            unsafe
            {
                //byte[] PUConfig = new byte[] { 0x32, 0x05, 0x20, 0x01, 0x2a, 0x01, 0x30, 0x38, 0x01 }; //gpu
                byte[] PUConfig = new byte[] { 0x0a, 0x07, 0x0a, 0x03, 0x67, 0x70, 0x75, 0x10, 0x00 }; //cpu
                fixed (void* ptr = &PUConfig[0])
                {
                    options.SetConfig(new IntPtr(ptr), PUConfig.Length);
                }
            }
            TFSession session;
            var graph = new TFGraph();
            using (TFSession sess= new TFSession(graph,options))     
            using (var metaGraphUnused = new TFBuffer())
            {
                session=sess.FromSavedModel(options, null, "shuffle2",new[] { "serve" }, graph, metaGraphUnused);
                IEnumerable<TensorFlow.DeviceAttributes> iem = session.ListDevices();
                foreach (object obj in iem)
                {
                    Console.WriteLine(((DeviceAttributes)obj).Name);
                }
                var labels = File.ReadAllLines("shuffle2/label.txt");
                //打印节点名称
                /*IEnumerable<TensorFlow.TFOperation> iem = graph.GetEnumerator();
                foreach (object obj in iem)
                {
                    Console.WriteLine(((TFOperation)obj).Name);
                }*/
                //while(true)
                float[] eimg = new float[224 * 224*3];
                for (int i = 0; i < 224 * 224*3; i++)
                    eimg[i] = 0;
                TFShape tfs = new TFShape(1, 224, 224, 3);
                TFTensor ten = TFTensor.FromBuffer(tfs, eimg, 0, 224 * 224 * 3);
                for (int j=0;j<3;j++)
                {
                    var runner = session.GetRunner();
                    runner.AddInput(graph["image"][0], ten).Fetch(graph["classes"].Name);
                    var output = runner.Run();
                }
                ten.Dispose();
                string[] files = Directory.GetFiles("images", "*.*");
                //while(true)
                foreach (string file in files)
                {
                    //byte[] contents = File.ReadAllBytes(file);
                    DateTime bft = DateTime.Now;
                    //var tensor = ImageUtil.CreateTensorFromImageFile(file);
                    TFTensor tensor = TFTensor.FromBuffer(tfs, eimg, 0, 224 * 224 * 3);
                    var runner = session.GetRunner();
                    runner.AddInput(graph["image"][0], tensor).Fetch(graph["classes"].Name);
                    var output = runner.Run();
                    DateTime aft = DateTime.Now;
                    TimeSpan ts = aft.Subtract(bft);
                    System.Threading.Thread.Sleep(50);
                    var result = output[0];
                    int class_ = ((int[])result.GetValue(jagged: true))[0];
                    Console.WriteLine(file + " best_match: " + class_ + " " + labels[class_] + " time: " + ts.TotalMilliseconds);
                }
            }
		}
	}
}
