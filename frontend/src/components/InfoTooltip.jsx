import { useState } from 'react';

/**
 * 물음표 아이콘 hover 시 한글+영어 설명을 보여주는 툴팁.
 * @param {{ ko: string, en: string }} props
 */
export default function InfoTooltip({ ko, en }) {
  const [show, setShow] = useState(false);

  return (
    <span
      className="relative inline-flex items-center ml-1 cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span className="w-4 h-4 rounded-full bg-gray-600 text-gray-300 text-[10px] flex items-center justify-center font-bold">
        ?
      </span>
      {show && (
        <div className="absolute z-50 bottom-6 left-1/2 -translate-x-1/2 w-72 bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl text-xs leading-relaxed">
          <p className="text-gray-200 mb-2">{ko}</p>
          <p className="text-gray-400 italic">{en}</p>
          <div className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-800 border-r border-b border-gray-600 rotate-45 -mt-1" />
        </div>
      )}
    </span>
  );
}
