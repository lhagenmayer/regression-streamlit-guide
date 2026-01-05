# Visual Guide: Perplexity AI Interpretation Feature

## User Interface Overview

This document describes the visual layout and user interaction flow for the new AI interpretation feature.

## Layout Structure

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Linear Regression Guide                                 │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Parameter Sidebar                                                          │
│  ┌──────────────────┐                                                       │
│  │ 🎛️ Parameter     │                                                       │
│  │                  │                                                       │
│  │ 📊 Datensatz     │                                                       │
│  │ [dropdown]       │                                                       │
│  │                  │                                                       │
│  └──────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Main Content Area                                                          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌────────────────────────────┬──────────────────────────────────────────┐ │
│  │  LEFT COLUMN (60%)         │  RIGHT COLUMN (40%)                      │ │
│  │                            │                                          │ │
│  │  ### 📊 R Output           │  📖 Erklärung der R Output Abschnitte   │ │
│  │  (Automatisch aktualisiert)│  [expandable section]                   │ │
│  │                            │                                          │ │
│  │  ┌──────────────────────┐  │  ───────────────────────────────────────│ │
│  │  │                      │  │                                          │ │
│  │  │  [R Output Plot]     │  │  ### 🤖 AI-Interpretation               │ │
│  │  │                      │  │                                          │ │
│  │  │  Shows:              │  │  ┌────────────────────────────────────┐ │ │
│  │  │  - Call              │  │  │  🔍 Interpretation generieren     │ │ │
│  │  │  - Residuals         │  │  └────────────────────────────────────┘ │ │
│  │  │  - Coefficients      │  │  [Primary button, full width]          │ │
│  │  │  - Model stats       │  │                                          │ │
│  │  │                      │  │  [After clicking button:]                │ │
│  │  └──────────────────────┘  │                                          │ │
│  │                            │  #### 📝 Interpretation:                │ │
│  │                            │  [AI-generated text in German]          │ │
│  │                            │                                          │ │
│  │                            │  _Generiert von Perplexity AI_          │ │
│  │                            │                                          │ │
│  │                            │  ▶ 📋 An AI gesendete Daten anzeigen   │ │
│  │                            │  [expandable section]                   │ │
│  │                            │                                          │ │
│  │                            │  ┌─────────────┬─────────────┐          │ │
│  │                            │  │ 💾 Download │  💡 Tipp    │          │ │
│  │                            │  └─────────────┴─────────────┘          │ │
│  │                            │                                          │ │
│  │                            │  [Text area with prompt]                │ │
│  │                            │                                          │ │
│  │                            │  🔄 Neue Interpretation                 │ │
│  │                            │  [button]                               │ │
│  └────────────────────────────┴──────────────────────────────────────────┘ │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  [Tabs: 📈 Einfache Regression | 📊 Multiple Regression | 📚 Datensätze] │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Interpretation Button (Before Click)

```
┌─────────────────────────────────────────────┐
│   ### 🤖 AI-Interpretation                  │
│                                             │
│   ┌───────────────────────────────────────┐ │
│   │  🔍 Interpretation generieren        │ │
│   └───────────────────────────────────────┘ │
│   [Primary blue button, full width]        │
└─────────────────────────────────────────────┘
```

### 2. Loading State (During API Call)

```
┌─────────────────────────────────────────────┐
│   ### 🤖 AI-Interpretation                  │
│                                             │
│   ⏳ 🤔 Analysiere Modell mit Perplexity   │
│   AI...                                     │
│   [Spinner animation]                       │
└─────────────────────────────────────────────┘
```

### 3. Interpretation Display (After Success)

```
┌──────────────────────────────────────────────────────────┐
│   ### 🤖 AI-Interpretation                               │
│                                                          │
│   #### 📝 Interpretation:                                │
│                                                          │
│   **1. Modellqualität**                                  │
│   Das Modell zeigt eine sehr gute Anpassung mit einem   │
│   R² von 0.9175, was bedeutet, dass 91.75% der Varianz  │
│   in der Zielvariable durch die Prädiktoren erklärt...  │
│                                                          │
│   **2. Koeffizienten-Interpretation**                    │
│   Der Intercept beträgt 2.12 und ist hochsignifikant... │
│   ...                                                    │
│                                                          │
│   _Generiert von Perplexity AI_                          │
│                                                          │
│   ▶ 📋 An AI gesendete Daten anzeigen                   │
│   [Collapsed expandable section]                        │
│                                                          │
│   ┌────────────────────────────────────────────┐        │
│   │  🔄 Neue Interpretation                    │        │
│   └────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 4. Prompt Data Section (Expanded)

```
┌──────────────────────────────────────────────────────────┐
│   ▼ 📋 An AI gesendete Daten anzeigen                   │
│                                                          │
│   **Dieser Prompt wurde an die Perplexity API gesendet:**│
│                                                          │
│   ┌────────────────────┬─────────────────────────────┐  │
│   │ 💾 Als Datei       │ 💡 Tipp: Text unten         │  │
│   │ herunterladen      │ auswählen & kopieren        │  │
│   └────────────────────┴─────────────────────────────┘  │
│                                                          │
│   ┌────────────────────────────────────────────────┐    │
│   │ Analysiere bitte folgendes Regressionsmodell  │    │
│   │ und gib eine verständliche Interpretation in  │    │
│   │ deutscher Sprache:                             │    │
│   │                                                │    │
│   │ **Modellübersicht:**                           │    │
│   │ - Modelltyp: Linear Regression                 │    │
│   │ - Anzahl Beobachtungen (n): 50                 │    │
│   │ - R²: 0.9175 (91.75% der Varianz erklärt)     │    │
│   │ ...                                            │    │
│   │                                  [Scrollable]  │    │
│   └────────────────────────────────────────────────┘    │
│   [Text area, 300px height, full width]                 │
└──────────────────────────────────────────────────────────┘
```

### 5. Error State

```
┌─────────────────────────────────────────────┐
│   ### 🤖 AI-Interpretation                  │
│                                             │
│   ❌ Fehler bei der API-Anfrage: Invalid   │
│   API key                                   │
│                                             │
│   ┌───────────────────────────────────────┐ │
│   │  🔄 Erneut versuchen                 │ │
│   └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### 6. Not Configured State

```
┌──────────────────────────────────────────────────────┐
│   ### 🤖 AI-Interpretation                           │
│                                                      │
│   ⚠️ Perplexity API nicht konfiguriert.             │
│   Setzen Sie die Umgebungsvariable                  │
│   `PERPLEXITY_API_KEY` um diese Funktion zu nutzen. │
│                                                      │
│   ▶ ℹ️ Wie konfiguriere ich die API?                │
│   [Expandable with setup instructions]              │
└──────────────────────────────────────────────────────┘
```

## Color Scheme

- **Primary Button**: Blue (`type="primary"`)
- **Warning**: Yellow/Orange background
- **Error**: Red text with light red background
- **Info**: Blue background
- **Success**: Green checkmark
- **Text**: Default dark gray/black
- **Code/Prompt**: Light gray background, monospace font

## Interactive Elements

### Button States

1. **Interpretation generieren** (Primary)
   - Default: Blue, white text
   - Hover: Darker blue
   - Click: Triggers API call

2. **Neue Interpretation** (Secondary)
   - Default: Gray outline
   - Hover: Light gray background
   - Click: Clears current interpretation

3. **Erneut versuchen** (Secondary)
   - Same as "Neue Interpretation"
   - Only shown on error

4. **Als Datei herunterladen** (Download button)
   - Icon: 💾
   - Downloads `.txt` file with prompt

### Expandable Sections

1. **"📋 An AI gesendete Daten anzeigen"**
   - Collapsed by default
   - Shows full prompt when expanded
   - Includes download and copy options

2. **"ℹ️ Wie konfiguriere ich die API?"**
   - Only shown when API not configured
   - Contains setup instructions

## Responsive Behavior

### Desktop (>1200px)
- R Output: 60% width
- Interpretation: 40% width
- Side-by-side layout

### Tablet (768px - 1200px)
- R Output: 55% width
- Interpretation: 45% width
- Buttons maintain full width within column

### Mobile (<768px)
- Columns stack vertically
- R Output shown first (top)
- Interpretation shown below
- All elements full width

## User Interaction Flow

### Happy Path

1. User loads page → sees R output automatically
2. User scrolls to right column → sees interpretation section
3. User clicks "🔍 Interpretation generieren"
4. Loading spinner appears (2-5 seconds)
5. Interpretation displays with formatted text
6. User reads interpretation
7. [Optional] User expands "📋 An AI gesendete Daten anzeigen"
8. [Optional] User downloads or copies prompt
9. [Optional] User clicks "🔄 Neue Interpretation" to regenerate

### Error Path

1. User loads page without API key configured
2. User sees warning: "⚠️ Perplexity API nicht konfiguriert"
3. User expands "ℹ️ Wie konfiguriere ich die API?"
4. User follows setup instructions
5. User reloads page
6. User continues with happy path

### Network Error Path

1. User clicks "🔍 Interpretation generieren"
2. API call fails (network error, invalid key, etc.)
3. Error message displays: "❌ Fehler bei der API-Anfrage: [error]"
4. User clicks "🔄 Erneut versuchen"
5. Goes back to step 1

## Accessibility Features

- All buttons have descriptive labels
- Icons used in addition to text (not alone)
- Error messages are clear and actionable
- Color is not the only indicator of state
- Text areas support keyboard selection
- Download button for users who can't copy text

## Performance Considerations

- API call: 2-5 seconds typical
- Loading state prevents multiple clicks
- Session state stores result (no re-fetch on re-render)
- Prompt cached in session state
- No automatic API calls (user-triggered only)

## Localization

- All UI text in German
- Prompt sent to API in German
- Response received in German
- Code comments in English (for developers)
- Documentation in English (for developers)
