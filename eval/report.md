# Informe de evaluación

Preguntas: 12 por sistema.

## Sistema agéntico contra RAG lineal

| Grupo | N | Correctas | Puntaje (1-5) | Alucinación | Fuente correcta | Latencia media | Tokens medios |
|---|---|---|---|---|---|---|---|
| Agéntico | 12 | 83% | 4.67 | 0% | 92% | 7.0 s | 3257 |
| RAG lineal | 12 | 92% | 4.67 | 0% | 92% | 8.8 s | 1418 |

## Por categoría

| Grupo | N | Correctas | Puntaje (1-5) | Alucinación | Fuente correcta | Latencia media | Tokens medios |
|---|---|---|---|---|---|---|---|
| Agéntico · docs | 12 | 83% | 4.67 | 0% | 92% | 7.0 s | 3257 |
| RAG lineal · docs | 12 | 92% | 4.67 | 0% | 92% | 8.8 s | 1418 |

## Precisión del supervisor

Ruta correcta: 12/12 (100%).

| Esperada | Elegida | Preguntas |
|---|---|---|
| docs | docs | d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12 |

Preguntas que pasaron de docs a la web por falta de contexto: ninguna.

## Casos fallidos del sistema agéntico

### d04 · ¿Cómo se agrega un subgrafo como nodo de un grafo padre?
- Ruta esperada/elegida: docs / docs (agente: docs)
- Juez: 3/5, Falta mencionar la alternativa cuando los esquemas de estado son distintos, por lo que la respuesta no cubre completamente la referencia.
- Fuentes citadas: langgraph-use-subgraphs.md, langgraph-use-subgraphs.md, langgraph-use-subgraphs.md; esperadas: langgraph-use-subgraphs.md
- Respuesta: Para agregar un subgrafo como nodo del grafo padre basta con compilar el subgrafo y pasarlo directamente a `add_node` del grafo padre; no se necesita una función wrapper porque el subgrafo leerá y escribirá en los mismos canales de estado que el padre[1][4].    Ejemplo:  ```python # Definir el subgr

### d07 · ¿Cómo se pausa un grafo para pedir aprobación humana y cómo se reanuda?
- Ruta esperada/elegida: docs / docs (agente: docs)
- Juez: 3/5, La respuesta menciona que el grafo se detiene con `interrupt()` y que el estado se guarda, lo cual coincide con la referencia. Sin embargo, omite el detalle clave de que la reanudación se realiza invocando de nuevo el grafo con `Command(resume=...)` usando el valor devuelto por `interrupt()`. Añade información extra sobre el checkpointer y thread_id que no está en la referencia pero no constituye alucinación. Por ello la respuesta es parcialmente correcta pero incompleta.
- Fuentes citadas: langgraph-thinking-in-langgraph.md; esperadas: langgraph-interrupts.md
- Respuesta: El grafo se detiene cuando llega a la llamada `interrupt()`. En ese momento se guarda todo el estado en el *checkpointer* y el flujo queda a la espera de la intervención humana. Más tarde, cuando se recibe la aprobación, el grafo se vuelve a ejecutar y retoma exactamente desde el punto donde se habí

## Detalle por pregunta

| Id | Sistema | Ruta | Correcta | Puntaje | Fuente | ms | Tokens |
|---|---|---|---|---|---|---|---|
| d01 | agentic | docs | sí | 5 | sí | 7862 | 3646 |
| d01 | linear | - | sí | 5 | sí | 2815 | 1525 |
| d02 | agentic | docs | sí | 5 | sí | 3884 | 2751 |
| d02 | linear | - | sí | 5 | sí | 3029 | 1453 |
| d03 | agentic | docs | sí | 5 | sí | 4812 | 3180 |
| d03 | linear | - | sí | 5 | sí | 2902 | 1352 |
| d04 | agentic | docs | no | 3 | sí | 15164 | 4457 |
| d04 | linear | - | no | 3 | sí | 13511 | 2024 |
| d05 | agentic | docs | sí | 5 | sí | 8289 | 3782 |
| d05 | linear | - | sí | 5 | sí | 13270 | 1676 |
| d06 | agentic | docs | sí | 5 | sí | 9454 | 3653 |
| d06 | linear | - | sí | 5 | sí | 11930 | 1491 |
| d07 | agentic | docs | no | 3 | no | 3975 | 2271 |
| d07 | linear | - | sí | 3 | no | 6837 | 885 |
| d08 | agentic | docs | sí | 5 | sí | 7098 | 2851 |
| d08 | linear | - | sí | 5 | sí | 7979 | 1132 |
| d09 | agentic | docs | sí | 5 | sí | 8074 | 3753 |
| d09 | linear | - | sí | 5 | sí | 12060 | 1460 |
| d10 | agentic | docs | sí | 5 | sí | 3606 | 2808 |
| d10 | linear | - | sí | 5 | sí | 12574 | 1502 |
| d11 | agentic | docs | sí | 5 | sí | 3894 | 2767 |
| d11 | linear | - | sí | 5 | sí | 10351 | 1378 |
| d12 | agentic | docs | sí | 5 | sí | 8028 | 3170 |
| d12 | linear | - | sí | 5 | sí | 8588 | 1144 |
