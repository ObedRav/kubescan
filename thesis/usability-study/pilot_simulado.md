# Prueba piloto del protocolo (simulación) — NO son datos de participantes reales

> **Naturaleza de este documento.** Antes del estudio con usuarios expertos se
> ejecutó un recorrido simulado del protocolo con tres perfiles (IA/ML,
> DevOps/SRE y QA de IaC) para **validar el protocolo y la herramienta**. Los
> valores de abajo son de esa simulación y **no sustituyen** a la evaluación con
> usuarios reales exigida por las instrucciones (§2.6). Sirven como *prueba
> piloto* metodológica: confirman que las tareas son ejecutables y claras y
> destapan incidencias a corregir antes de recrutar.

## Qué validó el piloto

- **Corrección funcional:** los tres directorios de muestra producen el veredicto
  esperado (`01_limpio`→CLEAN, `02_misconfig`→ISOLATED, `03_cadena`→ATTACK_CHAIN).
- **Determinismo:** puntuación *ensemble* idéntica y JSON byte a byte reproducible
  entre ejecuciones repetidas — propiedad clave para un control de calidad.
- **Ejecutabilidad de las tareas:** T1, T2, T4 y T5 se completan sin ayuda; T3
  requirió aclaración (incidencia corregida, ver abajo).

## Métricas simuladas (indicativas)

- SUS (simulado): **74,2 ± 2,9** (n=3) → «bueno», por encima del umbral 68.
- Éxito por tarea: T1/T2/T4/T5 100 %, T3 50 % (por la redacción, ya corregida).
- Aplicabilidad (media 1–5): A1 3,3 · A2 3,7 · A3 4,0 · A4 3,0.

## Incidencias de PROTOCOLO detectadas y corregidas

- **T3 presuponía una «flag de escape»** en `02_misconfig`, que por diseño no
  tiene ninguna (su riesgo viene de configuración/lateral). Además el directorio
  tenía un solo manifiesto, haciendo trivial el «de mayor riesgo». **Corregido:**
  se añadió un segundo manifiesto y se reformuló T3 a «la(s) flag(s) que explican
  su riesgo».
- **T1 ambiguo** respecto a si el participante instala o no. **Corregido:** vía
  autónoma (instalación desde cero) frente a vía rápida (entorno preparado).

## Hallazgos de HERRAMIENTA (mejoras de producto, pendientes de decisión)

Estas no bloquean el estudio, pero conviene valorarlas:

1. **Código de salida siempre 0**, incluso en ATTACK_CHAIN → no se puede *gate*
   una *pipeline* por `$?`; se propone una opción `--fail-on <veredicto|score>`.
2. **`InconsistentVersionWarning` de scikit-learn** en cada ejecución (modelos
   serializados con 1.6.1, entorno 1.7.2) → fijar la versión o reserializar.
3. **`clean_probability` ≈ 0** en todos los veredictos → confuso/engañoso en el
   informe; revisar o reetiquetar.
4. Menor: divergencia de vocabulario texto↔JSON («Escape fraction» vs
   `n_escape_capable`) y truncado de flags en la tabla `--show-nodes`.
