# Construcción y despliegue del formulario de evaluación

El cuestionario de la evaluación (perfil, resultado de tareas, SUS, aplicabilidad
y preguntas abiertas) se implementó como un **Google Form**. Para que el
instrumento fuera **reproducible y quedara bajo control de versiones**, no se
construyó a mano: se generó de forma programática mediante un script de
**Google Apps Script** (`crear_formulario.gs`), que crea todas las secciones y
preguntas en una única ejecución.

## Método de generación

El script se ejecutó una vez en Google Apps Script (https://script.google.com),
bajo la cuenta del autor, invocando la función `crearFormulario`. El script:

- crea el formulario con título y descripción (respuestas anónimas, uso
  exclusivamente académico);
- añade las cinco secciones descritas abajo, con los tipos de pregunta y las
  escalas correspondientes;
- **prefija los títulos** de los ítems (`SUS1`, `A1`, `T1`…) de modo que la
  exportación a CSV se corresponde directamente con las columnas de
  `plantilla_resultados.csv`;
- registra en el log la URL pública del formulario, que fue la distribuida a los
  participantes.

Al ser un script versionado, cualquier tercero puede regenerar el instrumento
íntegro sin reconstruirlo manualmente, lo que refuerza la reproducibilidad de la
evaluación.

## Estructura del formulario

| Sección | Contenido | Tipo de ítem |
|---------|-----------|--------------|
| 1. Perfil | rol, años con Kubernetes, familiaridad con seguridad, uso de CI/CD | opción única |
| 2. Tareas | T1–T5 | opción única (Logrado / con ayuda / No logrado) |
| 3. SUS | 10 enunciados estándar | escala lineal 1–5 |
| 4. Aplicabilidad | A1–A4 | escala lineal 1–5 |
| 5. Comentarios | más útil / fricción / caso de uso / otros | respuesta larga |

Los enunciados exactos de SUS y aplicabilidad son los de `GUIA_PARTICIPANTE.md`.

## Distribución

El enlace público del formulario se distribuyó a los participantes expertos junto
con la guía de tareas (`GUIA_PARTICIPANTE.md`), en **formato autoservicio
asíncrono** (~10 minutos por participante, sin sesión agendada), para minimizar
la carga sobre profesionales con disponibilidad limitada.

## Tratamiento de datos

Las respuestas se exportan de Google Forms a CSV; los valores se vuelcan en
`plantilla_resultados.csv` (una fila por participante) y se procesan con
`compute_results.py`, que calcula la puntuación **SUS** (ítems impares:
`valor − 1`; pares: `5 − valor`; suma × 2,5), la **tasa de éxito por tarea** y
las **medias de aplicabilidad**. El *scoring* es, por tanto, determinista y
auditable a partir de los datos crudos.
