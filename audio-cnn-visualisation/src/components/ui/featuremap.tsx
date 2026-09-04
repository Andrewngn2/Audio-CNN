import { getColor } from "~/lib/color";

const FeatureMap= ({data,title, internal, spectrogram}:
    {data: number[][];title: string; internal?: boolean; spectrogram?: boolean} ) => {
        if(!data || data.length === 0 || !data[0]?.length) return null;

        const firstRow = data[0];
        if(!firstRow || firstRow.length === 0) return null;

        const mapHeight = data.length;
        const mapWidth = firstRow.length;

        const absMax = data
        .flat()
        .reduce((acc, val)=> Math.max(acc,Math.abs(val ?? 0)) ,0);

        return(
        <div className= "w-full text-center">
            <svg viewBox={`0 0 ${mapWidth} ${mapHeight}`} preserveAspectRatio="none"
            className={`mx-auto blcok rounded border border-stone-200 ${internal ?"w-full max-w-32": spectrogram ? "w-full-object-contained":  "w-full max-w[500px] max-h-[300px] object-contain"}`}>
                {data.flatMap((row,i) => row.map((value,j)=> {
                    const normalizedValues = absMax === 0 ? 0 : value / absMax;
                    const [r,g,b] = getColor(normalizedValues);
                    return <rect key={`${i}-${j}`} x={j} y={i} width={1} height={1} fill={`rgb(${r}, ${g}, ${b})`} />;
                }),
                )}
            </svg>
            <p className="mt-1 text-xs text-stone-500">{title}</p>
            </div>)

    };

    export default FeatureMap;