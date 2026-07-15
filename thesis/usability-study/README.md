# Evaluación de usabilidad y aplicabilidad de kubescan — materiales

Materiales y documentación de la evaluación con usuarios expertos exigida para un
TFE de tipo *desarrollo de software* (instrucciones §2.6). Documentan **cómo se
diseñó, construyó y condujo** la evaluación, de forma reproducible.

## Contenido

| Archivo | Qué es |
|---------|--------|
| `protocolo_evaluacion.md` | Diseño del estudio: participantes, tareas, métricas (SUS + éxito + aplicabilidad), plan de análisis y amenazas a la validez. |
| `GUIA_PARTICIPANTE.md` | Material entregado a cada participante: tareas con comandos y enlace al cuestionario. |
| `FORMULARIO.md` | Cómo se construyó y desplegó el cuestionario (Google Form generado por script) y cómo se tratan los datos. |
| `crear_formulario.gs` | Script de Google Apps Script que genera el formulario de forma reproducible. |
| `muestras/` | Conjuntos de manifiestos de ejemplo usados en las tareas (verdad esperada: CLEAN / ISOLATED / ATTACK_CHAIN). |
| `plantilla_resultados.csv` | Plantilla de volcado de las respuestas (una fila por participante). |
| `compute_results.py` | Cálculo determinista de SUS, tasas de éxito y medias de aplicabilidad a partir del CSV. |
| `pilot_simulado.md` | Prueba piloto del **protocolo** (recorrido simulado que validó las tareas y detectó incidencias). No son datos de participantes reales. |

## Estado

El instrumento se construyó y distribuyó a los participantes expertos. La
recogida y el análisis de las respuestas reales, así como la redacción de la
sección de resultados en la memoria, se completan una vez recibidas.
