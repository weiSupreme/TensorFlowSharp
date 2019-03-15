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
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;

namespace ExampleInceptionInference
{
	class MainClass
	{
        static TFShape tfs = new TFShape(1, 224, 224, 1);
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
                session=sess.FromSavedModel(options, null, "tzb",new[] { "serve" }, graph, metaGraphUnused);
                IEnumerable<TensorFlow.DeviceAttributes> iem = session.ListDevices();
                foreach (object obj in iem)
                {
                    Console.WriteLine(((DeviceAttributes)obj).Name);
                }
                var labels = File.ReadAllLines("tzb/label.txt");
                //打印节点名称
                /*IEnumerable<TensorFlow.TFOperation> iem = graph.GetEnumerator();
                foreach (object obj in iem)
                {
                    Console.WriteLine(((TFOperation)obj).Name);
                }*/
                //while(true)
                //string[] files = Directory.GetFiles("tzb/images/C2", "*.*");
                string[] files = Directory.GetFiles("D:/实习/DL/jiu/train", "*.*");
                //while(true)
                //FileStream fs = new FileStream("res.txt", FileMode.Create);
                int cnt = 0;
                foreach (string file in files)
                {
                    //Console.WriteLine(file);
                    DateTime bft = DateTime.Now;
                    var tensor = Image2Tensor(file);
                    //DateTime bft = DateTime.Now;
                    //DateTime aft = DateTime.Now;
                    //break;
                    var runner = session.GetRunner();
                    runner.AddInput(graph["images"][0], tensor).Fetch(graph["classes"].Name);
                    var output = runner.Run();
                    DateTime aft = DateTime.Now;
                    TimeSpan ts = aft.Subtract(bft);
                   // System.Threading.Thread.Sleep(50);
                    var result = output[0];
                    int class_ = ((int[])result.GetValue(jagged: true))[0];
                    if (class_ == 1)
                    {
                        cnt++;
                        Console.WriteLine(file + " best_match: " + class_ + " " + labels[class_] + " time: " + ts.TotalMilliseconds+" number:"+ cnt.ToString());
                        //获得字节数组
                        //byte[] data = System.Text.Encoding.Default.GetBytes(Path.GetFileName(file) + "\n");
                        //开始写入
                        //fs.Write(data, 0, data.Length);
                    }
                }
                //清空缓冲区、关闭流
                //fs.Flush();
                //fs.Close();
                Console.Write("finish");
                Console.ReadLine(); //等待用户按一个回车
            }
		}

        static int cropSize=336;
        static int resizeSize = 224;
        static TFTensor Image2Tensor(string path)
        {
            //先截取感兴趣区域
            Bitmap img_ = new Bitmap(path);
            Rectangle rect = new Rectangle(145, 153, cropSize, cropSize);
            //Rectangle rect = new Rectangle(120, 50, cropSize, cropSize);
            Bitmap img = img_.Clone(rect, PixelFormat.Format8bppIndexed);
            BitmapData srcData = img.LockBits(new Rectangle(0,0,cropSize,cropSize), ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);
            System.IntPtr srcPtr = srcData.Scan0;
            byte[] cropArr = new byte[cropSize * cropSize];
            System.Runtime.InteropServices.Marshal.Copy(srcPtr, cropArr, 0, cropSize * cropSize);
            img.UnlockBits(srcData);
            TFShape tfs_ = new TFShape(cropSize, cropSize, 1);
            TFTensor tensor = TFTensor.FromBuffer(tfs_,cropArr,0,cropSize*cropSize);
            return tensor;
        }

        static TFTensor Image2Tensor2(string path)
        {
            //先截取感兴趣区域
            Bitmap img = new Bitmap(path);
            BitmapData srcData = img.LockBits(new Rectangle(0, 0, 224, 224), ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);
            System.IntPtr srcPtr = srcData.Scan0;
            byte[] cropArr = new byte[224 * 224];
            System.Runtime.InteropServices.Marshal.Copy(srcPtr, cropArr, 0, 224 * 224);
            img.UnlockBits(srcData);
            TFShape tfs_ = new TFShape(224, 224, 1);
            TFTensor tensor = TFTensor.FromBuffer(tfs_, cropArr, 0, 224 * 224);
            return tensor;
        }
    }
}
