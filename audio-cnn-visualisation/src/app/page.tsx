"use client"



import { Button, Progress } from "@base-ui/react";
import Link from "next/link";
import React from "react";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import ColorScale from "~/components/ui/ColorScale";
import FeatureMap from "~/components/ui/featuremap";
import Waveform from "~/components/ui/Waveform";

interface Prediction{
    class: string;
    confidence: number;
}

interface LayerData{
  shape: number[];
  values:number[][];
}

interface VizData{
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse{
  predictions: Prediction[];
  visualization: VizData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

function splitLayer(visualization: VizData) { 
  const main: [string, LayerData][] = []
  const internals: Record<string, [string, LayerData][]> = {};
  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name,data])
    } else{
      const [parent] = name.split(".");
      if (parent === undefined) continue;

      if(!internals[parent]) internals[parent] = [];
      internals[parent].push([name,data])
    }
}
  return {main, internals};
}


export default function HomePage() {
  const [vizData, setVizData] = React.useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [fileName, setFileName] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setIsLoading(true)
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try{
              const arrayBuffer = reader.result as ArrayBuffer;
      const base64String = btoa(new Uint8Array(arrayBuffer).reduce((data,byte) => data + String.fromCharCode(byte), ""));

      const response  = await fetch("https://andrewngn2--audio-cnn-inference-audioclassifier-inference.modal.run",
        {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ audio_data: base64String }),
        }
      );

      if (!response.ok) {
        throw new Error(`API error ${response.statusText}` );
      }

      const data: ApiResponse = await response.json();
      setVizData(data);


      } catch (err) {
        setError(err instanceof Error ? err.message: "An unknown error occurred");
      } finally {
        setIsLoading(false);
      }




    };
    reader.onerror = () => { 
      setError("Failed to read the file. Please try again.");
      setIsLoading(false);
  };
    };

  const {main, internals} = vizData ? splitLayer(vizData.visualization) : {main: [], internals: {}};

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[100%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900"> CNN Audio Visualizer

          </h1>
          <p className = "mb-8 text-lg text-stone-600">
            Upload a WAV file to see the model's predictions and feature maps
          </p>

          <div className= "flex flex-col items-center">
            <div className="relative inline-block">
              <input type = "file" 
              accept=".wav" 
              id="file-upload" 
              onChange= {handleFileChange}
              disabled={isLoading}
              className="absolute inset-0-w-full cursor pointer opacity-0">
              </input>
              <Button 
              className="border-stone-300"
              disabled={isLoading}>
                {isLoading ? "Analysing..." : "Choose File"}
              </Button>
            </div>


            {fileName && (<Badge className="mt-4 bg-stone-200 text-stone-700">{fileName}</Badge>)}
          </div>

        </div>

        {error && (<Card className ="mb-4 border-red-500 bg-red-50">
          <CardContent>
            <p 
            className="text-red-500">Error: {error}
              </p>
          </CardContent>
          </Card>)}

          {vizData && (<div className = "space-y-8">
            <Card>
              <CardHeader>
                Top Predictions
              </CardHeader>
              <CardContent>
                <div className= "space-y-3">
                  {vizData.predictions.slice(0,3).map((pred,i) => (<div key={pred.class} className ="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>
                        {pred.class.replaceAll("__"," ")}
                      </span>
                    </div>
                    <Badge variant= {i === 0 ? "default" : "secondary"}> {(pred.confidence * 100).toFixed(1)}%</Badge>
                    <Progress.Root value={pred.confidence * 100} className="h-2">
                      <Progress.Track className="h-full overflow-hidden rounded-full bg-stone-200">
                        <Progress.Indicator className="h-full bg-stone-900 transition-all" />
                      </Progress.Track>
                    </Progress.Root>
                  </div>))}
                </div>
              </CardContent>
            </Card>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card><CardHeader className= "text-stone-900">
                  <CardTitle>Input Spectorgram
              </CardTitle>
                   </CardHeader>
                   <CardContent>
                    <FeatureMap
                      data={vizData.input_spectrogram.values}
                      title={vizData.input_spectrogram.shape.join("x")}
                      spectrogram
                    />
                    <div className="mt-5 flex justify-end">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                    </div>
                   </CardContent>
                   </Card>
                   <Card><CardHeader>
                    <CardTitle>Audio Waveform
              </CardTitle>
                   </CardHeader>

                   <CardContent>
                    <Waveform data ={vizData.waveform.values} title={`${vizData.waveform.duration.toFixed(2)}s 8 ${vizData.waveform.sample_rate}Hz`} />
                   </CardContent>
                   </Card>
            </div>
            {}
            <Card><CardHeader><CardTitle>Convolutional Layer Outputs
              </CardTitle></CardHeader>
              <CardContent>
                <div className ="grid grid-cols-5 gap-6">
                  {main.map(([mainName, mainData])=> (
                    <div key={mainName} className= "space-y-4">
                      <div>
                        <h4 className="mb-2 font medium text-stone-700">{mainName}

                        </h4>
                        <FeatureMap data={mainData.values} title={`${mainData.shape.join("x")}`} />
                      </div>

                      {internals[mainName] && (
                        <div className="h-80 overflow-y-auto rounded border border-stone-200 bg-stone-50 p-2">
                          <div className="space-y-2">
                            {internals[mainName].sort(([a],[b]) => a.localeCompare(b))
                            .map(([layerName, LayerData]) =>(
                            <FeatureMap 
                            key ={layerName}
                            data ={LayerData.values} 
                            title={layerName.replace(`${mainName}.`, "")} 
                            internal= {true}
                           />
                           ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="mt-5 flex justify-end">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                    </div>
              </CardContent>
              </Card>

          </div> )}

      </div>
    </main>
  );
}
