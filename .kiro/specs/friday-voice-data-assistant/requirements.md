# Requirements Document

## Introduction

Friday is a proof-of-concept web app: a chat and voice assistant for pulling and analyzing US economic data. Friday is presented with the tagline "Your next-gen eco-buddy", where "eco" means economics, and the Browser_UI uses a futuristic dark theme with neon accents. The user ("Boss") types or speaks requests such as "please pull inflation." Friday fetches the series from FRED, shows a preview table and a downloadable CSV in the chat window, and narrates progress by voice. Friday then offers to run descriptive statistics or plot the data.

OpenAI GPT-5.6 Terra, hosted on Amazon Bedrock and called through the Bedrock Converse API, orchestrates the conversation. The Agent's tool-calling loop runs on the Microsoft Agent Framework harness. Amazon Transcribe handles speech-to-text and Amazon Polly handles text-to-speech. The Backend calls all three AWS services per request with the same AWS credentials, so there is no AWS infrastructure to provision. Friday runs locally on the user's machine.

The POC covers 10 well-known monthly FRED series. Data comes from the FRED API one series per request; the FRED-MD bulk dataset files are not used. Every data operation (fetching, descriptive statistics, filling missing values, plotting) runs as a deterministic, pre-written tool. The LLM picks which tool to call and narrates the results. The LLM never generates data, statistics, or chart code.

## Glossary

- **Friday_App**: The whole web application, including the Browser_UI and the Backend.
- **Browser_UI**: The client-side interface running in the user's web browser. It includes the Chat_Window, Chat_Input, Mic_Button, and Voice_Orb.
- **Chat_Window**: The scrollable pane showing the message history between the user and Friday. It can contain text, Preview_Tables, CSV download links, and Chart_Images.
- **Chat_Input**: The text box where the user types messages.
- **Mic_Button**: The control the user clicks to start and stop voice capture.
- **Voice_Orb**: The circular animated element in a side pane next to the Chat_Window. It shows the Assistant_State.
- **Assistant_State**: One of `idle`, `listening`, `working`, or `speaking`.
- **Backend**: The server-side component. It runs locally, holds Credentials, hosts the Agent and Tools, and proxies calls to Amazon Bedrock, Amazon Transcribe, Amazon Polly, and FRED.
- **Agent**: The Backend component that sends conversation turns to the Bedrock LLM, carries out the Tool calls the LLM requests, and returns Friday's responses. The Agent is built on the Agent_Harness.
- **Agent_Harness**: The Microsoft Agent Framework harness agent created with `create_harness_agent`. It drives the model calls, function (Tool) invocation, and per-Session conversation state for the Agent.
- **Bedrock_LLM**: OpenAI GPT-5.6 Terra, the large language model the Agent calls through the Amazon Bedrock runtime Converse API. It is identified by the Model_ID. Bedrock is serverless: the Backend calls the model per request with SigV4-signed requests, and nothing is provisioned.
- **Model_ID**: The Bedrock model identifier, read from configuration. The suggested value is `us.openai.gpt-5.6-terra`, the US cross-Region inference profile for GPT-5.6 Terra on the Bedrock runtime Converse API.
- **STT_Service**: Amazon Transcribe, the AWS speech-to-text service that converts recorded user audio into text.
- **TTS_Service**: Amazon Polly, the AWS text-to-speech service that converts Friday's Spoken_Text into audio.
- **AWS_Credential_Error**: A failure of a Bedrock_LLM, STT_Service, or TTS_Service call caused by expired, invalid, or insufficiently permitted AWS credentials, for example the AWS error codes `ExpiredToken`, `ExpiredTokenException`, `UnrecognizedClientException`, `InvalidSignatureException`, or `AccessDeniedException`.
- **Spoken_Text**: The part of a Friday response that is sent to the TTS_Service and played aloud.
- **Tool**: A deterministic, pre-written Backend function the Agent can invoke. The Tools are Fetch_Tool, Stats_Tool, Fill_Tool, and Plot_Tool.
- **Fetch_Tool**: The Tool that retrieves an Indicator's data from the FRED_API and produces a Dataset.
- **Stats_Tool**: The Tool that computes Descriptive_Statistics for a Dataset.
- **Fill_Tool**: The Tool that fills missing values in a Dataset using a Fill_Method.
- **Plot_Tool**: The Tool that renders one or more Datasets as a line chart and produces a Chart_Image.
- **FRED_API**: The Federal Reserve Economic Data web API from the Federal Reserve Bank of St. Louis.
- **Indicator**: A named economic measure the user can request. The supported Indicators are defined in the Indicator_Map.
- **Indicator_Map**: A fixed, configurable mapping from each Indicator name (and its accepted aliases) to a FRED series ID and an optional Transformation.
- **Default_Indicator_Map**: The Indicator_Map the Backend uses when the Indicator_Map file path setting is unset. It contains exactly the 10 monthly Indicators in the table below. Each alias belongs to exactly one Indicator.

  | Indicator | Aliases | FRED series ID | Transformation |
  |---|---|---|---|
  | `cpi` | `consumer price index`, `cpi-u`, `headline cpi` | `CPIAUCSL` | none |
  | `inflation` | `inflation rate`, `cpi inflation`, `yoy inflation` | `CPIAUCSL` | YoY_Transformation |
  | `core cpi` | `core consumer price index`, `cpi less food and energy` | `CPILFESL` | none |
  | `unemployment rate` | `unemployment`, `jobless rate` | `UNRATE` | none |
  | `nonfarm payrolls` | `payrolls`, `nfp`, `total nonfarm employment` | `PAYEMS` | none |
  | `fed funds rate` | `federal funds rate`, `fed funds`, `effective federal funds rate` | `FEDFUNDS` | none |
  | `10-year treasury yield` | `10 year treasury yield`, `10-year yield`, `10y yield`, `ten-year treasury yield` | `GS10` | none |
  | `housing prices` | `home prices`, `house prices`, `case-shiller`, `case-shiller index` | `CSUSHPINSA` (S&P CoreLogic Case-Shiller U.S. National Home Price Index) | none |
  | `housing starts` | `new housing starts`, `housing units started` | `HOUST` | none |
  | `industrial production` | `industrial production index`, `ip index` | `INDPRO` | none |
- **YoY_Transformation**: The year-over-year percent change for monthly data: `value_t = (CPI_t / CPI_{t-12} − 1) × 100`.
- **Dataset**: A table held in session memory with a `date` column and one value column named after the Indicator. Each Dataset has a session-unique Dataset_ID.
- **Missing_Value**: A Dataset cell with no numeric value, including FRED's `.` placeholder.
- **Preview_Table**: An HTML table in the Chat_Window showing the first 10 rows of a Dataset.
- **CSV_Export**: A downloadable CSV file holding every row of a Dataset.
- **Descriptive_Statistics**: For each value column: count of non-missing values, count of Missing_Values, mean, standard deviation (sample, ddof = 1), minimum, 25th percentile, median, 75th percentile, maximum, first date, and last date.
- **Fill_Method**: One of `forward_fill` or `linear_interpolation`.
- **Chart_Image**: A raster image of a line chart produced by the Plot_Tool and shown inline in the Chat_Window.
- **Credentials**: The secret values the Backend uses: AWS access key ID, AWS secret access key, AWS session token (when set), and FRED API key. The same AWS credentials authenticate the Bedrock_LLM, STT_Service, and TTS_Service calls.
- **Session**: A single browser page load. Session state lives in Backend memory and is discarded when the Session ends.
- **Start_Command**: The single documented shell command that starts the Friday_App locally.
- **Local_URL**: The `http://127.0.0.1:<port>` address at which the Backend serves the Browser_UI.
- **App_Header**: The banner at the top of the Browser_UI page that shows the name "Friday" and the Tagline.
- **Tagline**: The exact text "Your next-gen eco-buddy", where "eco" means economics.
- **Dark_Theme**: The Browser_UI visual theme: a page background with a WCAG relative luminance of at most 0.05, light-colored text, and Neon_Accent colors.
- **Neon_Accent**: A saturated accent color from a fixed palette that includes at least one cyan/teal color and one violet color.
- **Reduced_Motion_Setting**: The browser's `prefers-reduced-motion: reduce` media query result.

## Requirements

### Requirement 1: Chat Interface and Text Input

**User Story:** As Boss, I want a chat window with a text box, so that I can type requests to Friday and see the conversation history.

#### Acceptance Criteria

1. THE Browser_UI SHALL display the Chat_Window, the Chat_Input, the Mic_Button, and a side pane containing the Voice_Orb together on a single page, with no navigation to another page needed to reach any of them.
2. WHEN the user presses the Enter key in the Chat_Input while the Chat_Input contains at least one non-whitespace character and no more than 2,000 characters, THE Browser_UI SHALL append the text to the Chat_Window as a user message and send the text to the Backend.
3. WHEN the Browser_UI appends submitted text to the Chat_Window as a user message, THE Browser_UI SHALL clear the Chat_Input.
4. IF the user presses the Enter key while the Chat_Input is empty or contains only whitespace characters, THEN THE Browser_UI SHALL leave the Chat_Window unchanged, send nothing to the Backend, and leave the Chat_Input contents unchanged.
5. WHEN the Backend returns a Friday response, THE Browser_UI SHALL append the response to the Chat_Window as a Friday message, in the order the Backend produced the responses, with a visible text label that tells Friday messages apart from user messages.
6. WHEN a new message is appended to the Chat_Window, THE Chat_Window SHALL scroll so that the newest message is fully visible.
7. IF the user presses the Enter key while the Chat_Input contains more than 2,000 characters, THEN THE Browser_UI SHALL send nothing to the Backend, keep the Chat_Input contents unchanged, and show a message indicating that the 2,000-character limit was exceeded.
8. IF the Backend returns an error or returns no response within 60 seconds of a user message being sent, THEN THE Browser_UI SHALL append a Friday message to the Chat_Window indicating that the request failed, and keep all earlier messages in the Chat_Window unchanged.
9. WHILE the Browser_UI is waiting for a Backend response to a sent user message, THE Browser_UI SHALL allow typing in the Chat_Input and SHALL ignore Enter-key submissions, sending nothing to the Backend until the response, error, or 60-second timeout arrives.

### Requirement 2: Voice Orb State Indicator

**User Story:** As Boss, I want a glowing orb that reacts when Friday is listening, working, or speaking, so that I can tell what Friday is doing at a glance.

#### Acceptance Criteria

1. THE Voice_Orb SHALL show exactly one Assistant_State at any time.
2. WHILE the Assistant_State is `idle`, THE Voice_Orb SHALL display a static appearance with no change in size, brightness, or glow.
3. WHILE the Assistant_State is `listening`, THE Voice_Orb SHALL display a glow whose intensity (size or brightness) rises as the microphone input level rises, is at its minimum during silence, and updates at least 10 times per second.
4. WHILE the Assistant_State is `working`, THE Voice_Orb SHALL display a glow that pulses at a fixed period between 1 and 2 seconds and does not depend on any audio level.
5. WHILE the Assistant_State is `speaking`, THE Voice_Orb SHALL display a glow whose intensity (size or brightness) rises as the playback audio level rises, is at its minimum during silent playback, and updates at least 10 times per second.
6. WHEN the Assistant_State changes, THE Voice_Orb SHALL switch to the new state's appearance within 200 ms.
7. WHEN the user submits a typed message from the Chat_Input, or stops voice capture with the Mic_Button, THE Browser_UI SHALL set the Assistant_State to `working`.
8. WHEN Friday's audio playback ends, or the Browser_UI receives a Backend response that has no Spoken_Text, and no other request to the Backend is pending, THE Browser_UI SHALL set the Assistant_State to `idle`.
9. WHEN the user starts voice capture with the Mic_Button and the browser grants microphone access, THE Browser_UI SHALL set the Assistant_State to `listening`.
10. WHEN audio playback of Friday's Spoken_Text starts, THE Browser_UI SHALL set the Assistant_State to `speaking`.
11. IF microphone access is denied, the Backend returns an error, or the STT_Service or TTS_Service fails, THEN THE Browser_UI SHALL set the Assistant_State to `idle` within 200 ms of the failure.
12. WHEN a Session starts, THE Browser_UI SHALL set the Assistant_State to `idle`.

### Requirement 3: Voice Input

**User Story:** As Boss, I want to speak my requests, so that I can talk to Friday hands-free instead of typing.

#### Acceptance Criteria

1. WHEN the user clicks the Mic_Button while the Assistant_State is `idle` or `speaking`, THE Browser_UI SHALL stop any Friday audio playback, start recording microphone audio, and set the Assistant_State to `listening` within 500 ms of the click.
2. WHEN the user clicks the Mic_Button while the Assistant_State is `listening`, THE Browser_UI SHALL stop recording, set the Assistant_State to `working`, and send the recorded audio to the Backend.
3. WHEN the Backend receives recorded audio, THE Backend SHALL send the audio to the STT_Service with the configured Transcribe language code and return the transcribed text to the Browser_UI, waiting no longer than 30 seconds for the STT_Service response and storing the audio in no AWS storage service (such as Amazon S3).
4. WHEN the Browser_UI receives transcribed text containing at least one non-whitespace character, THE Browser_UI SHALL trim leading and trailing whitespace and treat the trimmed text exactly like a submitted Chat_Input message (Requirement 1.2).
5. IF the user denies microphone permission or no microphone device is available, THEN THE Browser_UI SHALL display a message in the Chat_Window stating that microphone access is required for voice input and that typing remains available, and THE Browser_UI SHALL set the Assistant_State to `idle` without sending any audio to the Backend.
6. IF the STT_Service returns an error, returns text that is empty or whitespace-only, or does not respond within 30 seconds, or the Backend cannot be reached, THEN THE Browser_UI SHALL discard the recorded audio, display a message in the Chat_Window asking the user to repeat or type the request (or, for an AWS_Credential_Error, the message defined in Requirement 5.12), send no message to the Agent, and set the Assistant_State to `idle`.
7. WHILE the Assistant_State is `listening` or `working`, THE Browser_UI SHALL keep the Chat_Input enabled for typing.
8. WHILE the Assistant_State is `listening`, IF recording has continued for 60 seconds, THEN THE Browser_UI SHALL stop recording automatically and process the recorded audio as defined in criterion 2.
9. WHILE the Assistant_State is `listening`, IF the Browser_UI has detected speech in the recording and the input level then stays below the silence threshold for 2 continuous seconds, THEN THE Browser_UI SHALL stop recording automatically and process the recorded audio as defined in criterion 2.
10. WHILE the Assistant_State is `working`, WHEN the user clicks the Mic_Button, THE Browser_UI SHALL ignore the click, start no recording, and leave the Assistant_State unchanged.
11. WHILE the Assistant_State is `listening`, WHEN the user submits a typed Chat_Input message, THE Browser_UI SHALL stop recording, discard the recorded audio without sending it to the Backend, and process the typed message as defined in Requirement 1.2.

### Requirement 4: Friday Persona and Spoken Narration

**User Story:** As Boss, I want Friday to speak to me briefly about what it is doing and what it found, so that I can follow along without reading everything.

#### Acceptance Criteria

1. THE Agent SHALL refer to the assistant as "Friday" whenever the assistant names itself, SHALL refer to the user as "Boss" only when directly addressing them (such as a greeting, a question, an offer, or an error that needs the user's attention), and SHALL NOT require the word "Boss" in every Friday response.
2. THE Agent SHALL limit Spoken_Text to three kinds of content: status of actions started or completed (for example "Fetching inflation data, Boss." and "Done."), summaries and key findings in which every numeric value matches a value in a Tool result (rounding to at most 2 decimal places allowed), and the next actions Friday can perform.
3. THE Agent SHALL exclude full tables, CSV contents, raw data rows, and chart descriptions longer than two sentences from Spoken_Text.
4. WHEN the Agent starts a Fetch_Tool call, THE Friday_App SHALL begin playing a status message that names the Indicator being fetched before the Browser_UI displays the Preview_Table for that fetch.
5. WHEN a Tool call completes successfully, THE Friday_App SHALL speak a completion status message of no more than 10 words.
6. WHEN the Backend produces Spoken_Text, THE Backend SHALL send the Spoken_Text to the TTS_Service with the configured Polly voice ID and Polly engine and return the resulting audio to the Browser_UI.
7. THE Friday_App SHALL produce Spoken_Text for user messages that arrive by text input and by voice input.
8. IF the TTS_Service returns an error or the Backend receives no audio from the TTS_Service within 15 seconds, THEN THE Browser_UI SHALL display the Friday response text in the Chat_Window without audio, display a notice that audio is unavailable for that response (or, for an AWS_Credential_Error, the message defined in Requirement 5.12), and set the Assistant_State to `idle` if no request is in progress.
9. WHEN the Browser_UI receives Friday audio, THE Browser_UI SHALL set the Assistant_State to `speaking` and play the audio clips one at a time in the order the Backend produced them, with no two clips playing at the same time.
10. THE Agent SHALL limit the Spoken_Text of each Friday response to a maximum of 60 words.
11. IF a Tool call returns an error, THEN THE Agent SHALL include in the Spoken_Text a status message naming the action that failed and SHALL omit the completion status message for that Tool call.

### Requirement 5: LLM Orchestration via Amazon Bedrock

**User Story:** As Boss, I want an LLM on my AWS sandbox to understand my requests and call the right tools, so that I can ask for data in plain language.

#### Acceptance Criteria

1. WHEN the Backend receives a user message, THE Agent SHALL send the Bedrock_LLM identified by the Model_ID the Tool definitions and the full Session conversation history, consisting of every prior user message, Friday response, Tool call, and Tool result in the current Session followed by the new user message.
2. THE Backend SHALL read the Model_ID from configuration, with no Model_ID value hardcoded in source code, and SHALL call the Bedrock_LLM in the configured AWS region.
3. WHEN the Bedrock_LLM requests a Tool call with a defined Tool name and arguments that pass validation against that Tool's schema, THE Agent SHALL run the named Tool with the supplied arguments and return the Tool result to the Bedrock_LLM in the same turn.
4. THE Agent SHALL expose only the Fetch_Tool, Stats_Tool, Fill_Tool, and Plot_Tool to the Bedrock_LLM.
5. THE Agent SHALL produce every Dataset, Descriptive_Statistics result, filled Dataset, and Chart_Image only through Tool execution, and SHALL take no Dataset, Descriptive_Statistics value, or Chart_Image from Bedrock_LLM output text.
6. THE Agent SHALL instruct the Bedrock_LLM, in the instructions sent with every request, to use only numeric values that appear in Tool results when stating figures in Friday responses.
7. IF a single Bedrock_LLM request returns an error other than an AWS_Credential_Error or does not respond within 60 seconds, THEN THE Agent SHALL return a Friday response stating that the language model is unavailable, keep the Session conversation history recorded before that user message unchanged, and THE Browser_UI SHALL set the Assistant_State to `idle`.
8. IF the Bedrock_LLM requests a Tool name that is not one of the four defined Tools, or supplies arguments that fail validation against the Tool's schema, THEN THE Agent SHALL return to the Bedrock_LLM a validation error naming the invalid Tool name or argument without running any Tool, and SHALL count the rejected request toward the Tool call limit in criterion 9.
9. IF the Bedrock_LLM requests a 9th Tool call while handling a single user message, THEN THE Agent SHALL skip that Tool call, end the turn, and return a Friday response stating that the request could not be finished within the Tool call limit, keeping every Dataset and Chart_Image produced by the first 8 Tool calls in the Session.
10. WHEN the Bedrock_LLM returns a response that contains no Tool call, THE Agent SHALL return that response's text to the Browser_UI as the Friday response to the user message and add the response to the Session conversation history.
11. IF a Tool raises an error during execution, THEN THE Agent SHALL return to the Bedrock_LLM an error result indicating the Tool name and the failure reason, add no partial Dataset or Chart_Image from that Tool call to the Session, and continue the turn.
12. IF a Bedrock_LLM, STT_Service, or TTS_Service call fails with an AWS_Credential_Error, THEN THE Backend SHALL return to the Browser_UI a Friday response stating that the AWS credentials have expired or lack access and that the user should refresh the values in `.env` and restart the Backend, keep the Session conversation history recorded before that user message unchanged, and THE Browser_UI SHALL set the Assistant_State to `idle`.
13. WHEN the Backend reports or logs an AWS_Credential_Error, THE Backend SHALL exclude every Credential value, in full or in part, from the Friday response and from every log line (Requirement 12.4).
14. THE Agent SHALL run every conversation turn on the Agent_Harness, connected to the Bedrock_LLM through the Bedrock runtime Converse API.
15. THE Agent SHALL configure the Agent_Harness with the Fetch_Tool, Stats_Tool, Fill_Tool, and Plot_Tool as the only tools available to the Bedrock_LLM, and with the harness todo tracking, plan/execute modes, file memory, web search, tool auto-approval, compaction, skills, file access, shell execution, background agents, and looping capabilities disabled.

### Requirement 6: Indicator Resolution and Data Fetching

**User Story:** As Boss, I want to ask for indicators like inflation, CPI, unemployment, or housing prices by name, so that I can get the right data without knowing FRED series IDs.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Backend SHALL load the Indicator_Map from the file named by the Indicator_Map file path setting, or, if that setting is unset, use the Default_Indicator_Map of 10 monthly Indicators defined in the Glossary (Requirement 12.3), with startup failures for an invalid Indicator_Map file handled as defined in Requirement 12.7.
2. WHEN the Fetch_Tool receives an Indicator name or alias that matches an Indicator_Map entry after ignoring letter case and leading or trailing whitespace, THE Fetch_Tool SHALL request the mapped FRED series observations from the FRED_API.
3. WHEN the Fetch_Tool receives a start date, an end date, or both, each in YYYY-MM-DD format, THE Fetch_Tool SHALL limit the Dataset to observations whose dates fall on or after the start date and on or before the end date.
4. WHEN the Fetch_Tool receives neither a start date nor an end date, THE Fetch_Tool SHALL return every observation the FRED_API provides for the series.
5. WHEN the FRED_API returns observations, THE Fetch_Tool SHALL produce a Dataset sorted by ascending date, with one row per observation, a `date` column, and one numeric value column named after the Indicator, with every FRED `.` placeholder converted to a Missing_Value.
6. WHERE the Indicator_Map entry specifies the YoY_Transformation, THE Fetch_Tool SHALL apply the YoY_Transformation to the fetched series within the requested date range, exclude the first 12 observations of that range, and set the transformed value to a Missing_Value for any row where CPI_t or CPI_{t-12} is a Missing_Value.
7. WHERE the Indicator_Map entry specifies the YoY_Transformation, WHEN the Fetch_Tool fetches a monthly series whose untransformed Dataset for the requested date range contains more than 12 rows, THE Fetch_Tool SHALL produce a Dataset containing exactly 12 fewer rows than the untransformed Dataset for the same series and date range (metamorphic property).
8. WHEN the Fetch_Tool produces a Dataset, THE Backend SHALL store the Dataset in Session memory under a new Dataset_ID that differs from every existing Dataset_ID in the Session, and SHALL return to the Agent the Dataset_ID, Indicator name, FRED series ID, row count, first date, and last date.
9. IF the requested Indicator does not match any Indicator_Map entry under the matching rule in criterion 2, THEN THE Fetch_Tool SHALL return an error listing every supported Indicator name, and THE Agent SHALL relay those supported Indicator names to the user.
10. IF the FRED_API returns an error status, returns zero observations for the requested date range, or does not respond within 15 seconds of the request, THEN THE Fetch_Tool SHALL return an error identifying which of these three failures occurred, and THE Agent SHALL tell the user the fetch failed and offer to retry.
11. IF the Fetch_Tool returns an error, THEN THE Backend SHALL leave existing Session Datasets unchanged and SHALL create no Dataset, Preview_Table, or CSV_Export.
12. IF a supplied start date or end date is not a valid calendar date in YYYY-MM-DD format, or the start date is later than the end date, THEN THE Fetch_Tool SHALL return an error identifying the invalid date input without calling the FRED_API.
13. IF the YoY_Transformation leaves zero rows because the requested date range contains 12 or fewer observations, THEN THE Fetch_Tool SHALL return an error indicating the date range is too short for the YoY_Transformation.
14. THE Default_Indicator_Map SHALL contain exactly the 10 Indicators listed in the Glossary (`cpi`, `inflation`, `core cpi`, `unemployment rate`, `nonfarm payrolls`, `fed funds rate`, `10-year treasury yield`, `housing prices`, `housing starts`, `industrial production`), each mapped to a FRED series whose FRED-reported frequency is monthly.
15. THE Fetch_Tool SHALL obtain every Dataset from the FRED_API series observations for the single FRED series ID mapped to the requested Indicator.

### Requirement 7: Data Preview and CSV Download

**User Story:** As Boss, I want to see a preview of the fetched data and download the full dataset as a CSV, so that I can check the data and use it elsewhere.

#### Acceptance Criteria

1. WHEN the Fetch_Tool produces a Dataset, THE Browser_UI SHALL display in the Chat_Window a Preview_Table of the first 10 rows of the Dataset in ascending date order, along with the Indicator name, FRED series ID, total row count, and date range given as the first and last dates in YYYY-MM-DD format.
2. IF a Dataset contains between 1 and 9 rows, THEN THE Browser_UI SHALL display all rows of the Dataset in the Preview_Table.
3. WHEN the Browser_UI displays a Preview_Table, THE Browser_UI SHALL show dates in YYYY-MM-DD format and SHALL show each Missing_Value as an empty cell.
4. WHEN the Browser_UI displays a Preview_Table, THE Browser_UI SHALL display a download link for the CSV_Export of the same Dataset (identified by its Dataset_ID) directly beside the Preview_Table in the same chat message.
5. WHEN the user activates a CSV_Export download link, THE Backend SHALL return a CSV file with a `.csv` extension whose file name contains the FRED series ID, containing a header row (`date`, value column name) followed by every row of the Dataset in ascending date order, with dates in YYYY-MM-DD format, numeric values unrounded, and Missing_Values written as empty fields.
6. FOR ALL Datasets, parsing a CSV_Export SHALL produce a Dataset with the same column names, row count, dates, and values as the source Dataset, including Missing_Value positions (round-trip property).
7. WHEN the Fill_Tool produces a filled Dataset, THE Browser_UI SHALL display a new Preview_Table and CSV_Export download link for the filled Dataset under its own Dataset_ID, and SHALL keep the Preview_Table and CSV_Export download link for the source Dataset displayed and downloadable unchanged.
8. IF the user activates a CSV_Export download link whose Dataset_ID is not present in the current Session, THEN THE Backend SHALL return no file, and THE Browser_UI SHALL display an error message in the Chat_Window indicating the Dataset is no longer available and must be fetched again.
9. IF a Dataset contains 0 rows, THEN THE Browser_UI SHALL display a message in the Chat_Window indicating that no data rows were returned for the Indicator, in place of the Preview_Table and CSV_Export download link.

### Requirement 8: Next-Step Offer

**User Story:** As Boss, I want Friday to suggest what I can do next with the data, so that I can quickly move on to analysis.

#### Acceptance Criteria

1. WHEN the Fetch_Tool completes successfully, THE Agent SHALL end the same Friday response, in both the response text and the Spoken_Text, with a single question asking the user whether to run descriptive statistics or plot the data (for example, "Boss, do you want to run summary statistics, or do you want to plot the data?").
2. IF the Dataset produced by a successful Fetch_Tool call contains one or more Missing_Values, THEN THE Agent SHALL state in the same Friday response, in both the response text and the Spoken_Text, the count of Missing_Values and offer to fill them with the Fill_Tool, in addition to the question in criterion 1.
3. WHEN the Stats_Tool, Fill_Tool, or Plot_Tool completes successfully, THE Agent SHALL end the same Friday response, in both the response text and the Spoken_Text, by naming each action from the set {descriptive statistics, fill missing values, plot} other than the action just completed, and SHALL include "fill missing values" only if at least one loaded Dataset contains one or more Missing_Values.
4. WHEN the user accepts an offered action in the next turn without naming an Indicator or Dataset (for example, "plot it" or "yes, run the stats"), THE Agent SHALL invoke the matching Tool on the Dataset most recently produced in the Session, without asking the user to re-specify the Indicator.
5. IF a Fetch_Tool, Stats_Tool, Fill_Tool, or Plot_Tool call fails, THEN THE Agent SHALL omit the next-step offer from that Friday response.
6. IF the user requests descriptive statistics, filling, or plotting while the Session contains no Dataset, THEN THE Agent SHALL invoke no Tool, leave the Session state unchanged, and tell the user, in both the response text and the Spoken_Text, that no data is loaded and ask which Indicator to fetch.

### Requirement 9: Descriptive Statistics Tool

**User Story:** As Boss, I want summary statistics for a dataset, so that I can understand its distribution and range.

#### Acceptance Criteria

1. WHEN the Stats_Tool receives a Dataset_ID present in the Session, THE Stats_Tool SHALL compute the Descriptive_Statistics for the Dataset's value column, excluding Missing_Values from every statistic except the Missing_Value count, and SHALL report first date and last date as the dates of the earliest and latest rows holding a non-missing value.
2. THE Stats_Tool SHALL compute the 25th percentile, median, and 75th percentile using linear interpolation between the two closest ranks of the sorted non-missing values.
3. WHEN the Stats_Tool computes Descriptive_Statistics, THE Stats_Tool SHALL leave the Dataset's rows, values, and Dataset_ID unchanged.
4. WHEN the Stats_Tool returns Descriptive_Statistics, THE Browser_UI SHALL display a table in the Chat_Window with one row per Descriptive_Statistics item, each row showing the statistic's label and value, with numeric values other than counts rounded to 2 decimal places, counts shown as integers, and dates shown in YYYY-MM-DD format.
5. WHEN the Stats_Tool returns Descriptive_Statistics, THE Agent SHALL include in the Spoken_Text a summary of 1 to 4 findings, where every number spoken appears in the Stats_Tool or Fetch_Tool result for that Dataset (allowing rounding to at most 2 decimal places).
6. FOR ALL Datasets, running the Stats_Tool twice on the same Dataset SHALL return identical results (determinism property).
7. FOR ALL Datasets with at least one non-missing value, the Stats_Tool results SHALL satisfy minimum ≤ 25th percentile ≤ median ≤ 75th percentile ≤ maximum, and count of non-missing values + count of Missing_Values = Dataset row count (invariant property).
8. IF a Dataset has exactly 1 non-missing value, THEN THE Stats_Tool SHALL report standard deviation as undefined and compute the remaining statistics.
9. IF a Dataset has 0 non-missing values, THEN THE Stats_Tool SHALL report the non-missing count as 0, the Missing_Value count as the Dataset row count, and every other statistic as undefined, without returning an error.
10. IF the Stats_Tool receives a Dataset_ID not present in the Session, THEN THE Stats_Tool SHALL return an error naming the unknown Dataset_ID, and THE Agent SHALL tell the user in the Chat_Window and Spoken_Text that the dataset was not found, leaving all Session Datasets unchanged.

### Requirement 10: Fill Missing Values Tool

**User Story:** As Boss, I want to fill gaps in a dataset, so that I can analyze and plot a continuous series.

#### Acceptance Criteria

1. WHEN the Fill_Tool receives a Dataset_ID present in the Session and a valid Fill_Method, THE Fill_Tool SHALL produce a new Dataset with Missing_Values filled using the Fill_Method, store the new Dataset in Session memory under a new Dataset_ID with the same value column name as the source Dataset, and leave the source Dataset and its Dataset_ID unchanged.
2. WHEN the Fill_Tool receives a Dataset_ID present in the Session and no Fill_Method, THE Fill_Tool SHALL apply `forward_fill`.
3. WHEN the Fill_Tool applies `forward_fill`, THE Fill_Tool SHALL replace each Missing_Value with the nearest non-missing value at an earlier date in the same Dataset.
4. WHEN the Fill_Tool applies `linear_interpolation`, THE Fill_Tool SHALL replace each Missing_Value at date d that lies between an earlier non-missing value v_a at date d_a and a later non-missing value v_b at date d_b with v_a + (v_b − v_a) × (days from d_a to d) / (days from d_a to d_b), where v_a and v_b are the nearest non-missing values on each side of d.
5. IF a Missing_Value has no non-missing value at an earlier date (for `forward_fill`), or has no non-missing value at an earlier date or no non-missing value at a later date (for `linear_interpolation`), THEN THE Fill_Tool SHALL leave that Missing_Value unfilled in the new Dataset and include it in the count of unfilled values.
6. WHEN the Fill_Tool completes, THE Fill_Tool SHALL return to the Agent the source Dataset_ID, the new Dataset_ID, the Fill_Method applied, the row count, the count of filled values, and the count of unfilled values, where filled count + unfilled count equals the source Dataset's Missing_Value count.
7. FOR ALL Datasets and Fill_Methods, the filled Dataset SHALL have the same row count and the same dates in the same order as the source Dataset, and every non-missing source value SHALL be unchanged (invariant property).
8. FOR ALL Datasets and Fill_Methods, applying the Fill_Tool to an already-filled Dataset with the same Fill_Method SHALL produce a Dataset with identical dates and values and report a filled count of 0 (idempotence property).
9. IF the Fill_Tool receives a Dataset_ID not present in the Session, THEN THE Fill_Tool SHALL return an error naming the unknown Dataset_ID, create no new Dataset, and leave all Session Datasets unchanged.
10. IF the Fill_Tool receives a Fill_Method other than `forward_fill` or `linear_interpolation`, THEN THE Fill_Tool SHALL return an error naming the invalid Fill_Method and listing the two supported Fill_Methods, create no new Dataset, and leave all Session Datasets unchanged.
11. IF the source Dataset contains zero Missing_Values, THEN THE Fill_Tool SHALL produce and store a new Dataset identical in dates and values to the source Dataset and report a filled count of 0 and an unfilled count of 0.

### Requirement 11: Line Chart Plotting Tool

**User Story:** As Boss, I want to plot one or several series on a line chart in the chat, so that I can see trends visually.

#### Acceptance Criteria

1. WHEN the Plot_Tool receives a list of 1 to 5 distinct Dataset_IDs that each exist in the current Session, THE Plot_Tool SHALL render a single Chart_Image containing exactly one line series per Dataset_ID on a single set of axes.
2. THE Plot_Tool SHALL render the Chart_Image with dates on the horizontal axis, values on the vertical axis, a horizontal axis label indicating date, a vertical axis label naming the Indicator when one series is plotted or a generic value label when more than one series is plotted, and a chart title equal to the supplied title or, if no title is supplied, a title listing the plotted Indicator names.
3. WHEN the Plot_Tool renders more than one series, THE Plot_Tool SHALL include a legend that labels each series with its Indicator name, with one legend entry per series.
4. THE Plot_Tool SHALL render charts using pre-written chart code with fixed styling, SHALL accept only Dataset_IDs, an optional title of at most 100 characters, and an optional date range as arguments, and SHALL leave every plotted Dataset unchanged.
5. WHERE a date range is supplied as a start date and/or end date in YYYY-MM-DD format, THE Plot_Tool SHALL plot only observations whose date is on or after the start date and on or before the end date, treating an omitted bound as unbounded on that side.
6. WHEN the Plot_Tool returns a Chart_Image, THE Browser_UI SHALL display the Chart_Image inline in the Chat_Window as a Friday message, with alt text listing the plotted Indicator names and the first and last plotted dates.
7. WHEN the Plot_Tool renders a series containing Missing_Values, THE Plot_Tool SHALL show a gap in the line at each Missing_Value instead of connecting the adjacent non-missing points.
8. IF the Plot_Tool receives an empty list, a Dataset_ID that does not exist in the current Session, the same Dataset_ID more than once, more than 5 Dataset_IDs, or a title longer than 100 characters, THEN THE Plot_Tool SHALL return an error to the Agent identifying the invalid argument and SHALL produce no Chart_Image.
9. IF the supplied date range is not in YYYY-MM-DD format, has a start date later than its end date, or contains no non-missing observations for any requested Dataset, THEN THE Plot_Tool SHALL return an error to the Agent describing the date range problem and SHALL produce no Chart_Image.
10. IF chart rendering fails for any other reason, THEN THE Plot_Tool SHALL return an error to the Agent indicating that the chart could not be rendered, SHALL produce no Chart_Image, and SHALL leave every Dataset unchanged.

### Requirement 12: Configuration and Credential Handling

**User Story:** As Boss, I want my AWS credentials and FRED key kept on the server, so that they are never exposed in the browser.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Backend SHALL read the following settings from environment variables: the required settings (AWS access key ID `AWS_ACCESS_KEY_ID`, AWS secret access key `AWS_SECRET_ACCESS_KEY`, AWS region `AWS_REGION`, FRED API key `FRED_API_KEY`, and Model_ID `BEDROCK_MODEL_ID`) and the optional settings (AWS session token `AWS_SESSION_TOKEN`, Polly voice ID `POLLY_VOICE_ID`, Polly engine `POLLY_ENGINE` such as `neural`, Transcribe language code `TRANSCRIBE_LANGUAGE_CODE` with default `en-US`, Indicator_Map file path `INDICATOR_MAP_PATH`, and Backend port `FRIDAY_PORT` with default `8000`).
2. WHEN the Backend starts and a `.env` file exists in the project root, THE Backend SHALL load settings from that file, giving precedence to the process environment value for any variable set in both the `.env` file and the process environment.
3. WHEN an optional setting is unset or empty at startup, THE Backend SHALL use the default value documented for that setting in `.env.example`, and for an unset Indicator_Map file path SHALL use the Default_Indicator_Map defined in the Glossary.
4. THE Backend SHALL exclude Credential values, in full or in part, from every HTTP response, HTML page, script, and log line, including error details relayed from Amazon Bedrock, Amazon Transcribe, Amazon Polly, or FRED that are sent to the Browser_UI or written to logs.
5. THE Browser_UI SHALL send all Bedrock_LLM, STT_Service, TTS_Service, and FRED_API requests through the Backend, making zero network requests directly to Amazon Bedrock, Amazon Transcribe, Amazon Polly, or FRED_API hosts during a Session.
6. IF a required setting is unset, empty, or only whitespace at startup, THEN THE Backend SHALL exit with a non-zero exit code before accepting any HTTP request, and SHALL print an error message naming every missing environment variable in a single message without printing any Credential value.
7. IF the Indicator_Map file path is set and the file does not exist, cannot be read, cannot be parsed as an Indicator_Map, or contains an entry without a FRED series ID, THEN THE Backend SHALL exit with a non-zero exit code before accepting any HTTP request, and SHALL print an error message naming the file path and the cause of the failure (including the Indicator name of any entry missing a FRED series ID).
8. THE Friday_App SHALL provide a `.env.example` file that lists every required and optional environment variable from criterion 1, marks each entry as required or optional, contains only placeholder values with no real Credential values, and shows the default value for each optional setting.
9. THE Friday_App SHALL include `.env` in the version-control ignore file.
10. WHEN the Backend starts and no `.env` file exists in the project root, THE Backend SHALL read settings from the process environment only and continue startup.
11. THE Backend SHALL authenticate every Bedrock_LLM, STT_Service, and TTS_Service request with the same AWS credentials from configuration (access key ID, secret access key, and session token when set), signed with AWS Signature Version 4 for the configured AWS region.
12. THE `.env.example` file SHALL show `us.openai.gpt-5.6-terra` as the suggested Model_ID value and `us-east-2` as the suggested AWS region value.
13. THE `.env.example` file SHALL mark the AWS session token as optional and SHALL state that the AWS session token is required when using temporary (sandbox) AWS credentials.

### Requirement 13: Local Startup

**User Story:** As Boss, I want to start Friday on my own machine with one command, so that I can run the POC locally.

#### Acceptance Criteria

1. THE Friday_App SHALL provide a single documented Start_Command that, when run from the project root, installs only the declared project dependencies, starts the Backend, and serves the Browser_UI at the Local_URL.
2. THE Backend SHALL bind only to the 127.0.0.1 loopback address, so that the Browser_UI and Backend endpoints do not accept connections on any other network interface.
3. THE Backend SHALL listen on the port given by the Backend port setting (`FRIDAY_PORT`), or on port 8000 when that setting is unset, empty, or only whitespace.
4. WHEN the Backend has started successfully and is accepting HTTP requests, THE Backend SHALL print to the console the Local_URL of the Browser_UI, including the port number the Backend is actually listening on.
5. IF the Backend port setting is not an integer from 1 to 65535, or the port is already in use or cannot be bound, THEN THE Backend SHALL exit with a non-zero exit code before accepting any HTTP request and print an error message naming the port value and the cause of the failure (invalid value, port in use, or bind not permitted).
6. THE Friday_App SHALL provide a README that lists the prerequisites (the required language runtime and its minimum version, AWS credentials with access to the Bedrock_LLM, STT_Service, and TTS_Service, and a FRED API key), explains how to copy `.env.example` to `.env` and fill in the values, gives the Start_Command, states the Local_URL to open in the browser, and explains how to refresh expired sandbox AWS credentials in `.env` and restart the Backend.
7. IF installation of the declared project dependencies fails while the Start_Command runs, THEN THE Friday_App SHALL exit with a non-zero exit code without starting the Backend and print an error message indicating that dependency installation failed.
8. WHEN the Backend process receives an interrupt signal (Ctrl+C) in the console, THE Backend SHALL stop accepting HTTP requests, release the listening port, and exit within 5 seconds.

### Requirement 14: Branding and Visual Theme

**User Story:** As Boss, I want Friday to look futuristic and carry its eco-buddy identity, so that the POC feels like a next-gen assistant.

#### Acceptance Criteria

1. THE Browser_UI SHALL display the App_Header, containing the name "Friday" and the exact Tagline text, at the top of the same page as the Chat_Window.
2. THE Browser_UI SHALL set the HTML page title to a value that includes "Friday".
3. THE Browser_UI SHALL render the page in the Dark_Theme, using Neon_Accent colors for the Voice_Orb glow, a border or label on every Friday message in the Chat_Window, and the interactive controls (Chat_Input focus state and Mic_Button).
4. THE Browser_UI SHALL render App_Header text, Chat_Input text, chat message text, Preview_Tables, and Descriptive_Statistics tables with a text-to-background contrast ratio of at least 4.5:1 (WCAG 2.x AA for normal text).
5. THE Browser_UI SHALL load every font, script, and stylesheet from the Backend at the Local_URL, making zero requests to third-party hosts (including CDNs) for fonts, scripts, or stylesheets.
6. WHERE the Reduced_Motion_Setting is active, THE Voice_Orb SHALL display each of the 4 Assistant_States as a static color that differs from the colors of the other 3 states, with no size, brightness, or glow animation, in place of the animated appearances defined in Requirements 2.3, 2.4, and 2.5.
7. THE Plot_Tool SHALL apply, as the fixed styling defined in Requirement 11.4, a chart background with a WCAG relative luminance of at most 0.05, and axes, axis labels, tick labels, title, and legend text with a contrast ratio of at least 4.5:1 against that background.
8. THE Plot_Tool SHALL draw each line series in a Neon_Accent color, with every series in a single Chart_Image (up to the 5-series maximum of Requirement 11.1) drawn in a color different from every other series in that Chart_Image.

## Out of Scope

The following are excluded from this POC:

- User authentication, authorization, and multi-user support. The Backend is intended to run locally only (Requirement 13).
- Persistence of conversations, Datasets, or charts beyond a single Session.
- Deployment, hosting, containerization, and CI/CD.
- Provisioning or deploying any AWS infrastructure: no infrastructure as code, no hosted endpoints, and no storage buckets. Amazon Bedrock, Amazon Transcribe, and Amazon Polly are called per request, with nothing to deploy.
- Amazon Bedrock API keys (bearer tokens). The POC authenticates with AWS credentials (access key ID, secret access key, and session token).
- The Bedrock Mantle (OpenAI-compatible Chat Completions) endpoint. The POC calls the Bedrock_LLM only through the Bedrock runtime Converse API.
- Agent_Harness capabilities other than function (Tool) invocation and Session conversation state (see Requirement 5.15).
- Amazon Nova Sonic and other speech-to-speech models.
- Data sources other than the FRED_API, and Indicators not defined in the Indicator_Map.
- The FRED-MD bulk dataset files. Data comes from the FRED_API one series per request.
- Chart types other than a single-axes line chart (for example, dual axes, subplots, bar charts, or interactive charts).
- Statistical analysis beyond Descriptive_Statistics (for example, forecasting, regression, or correlation).
- Wake-word detection, continuous hands-free listening, and streaming (real-time) speech.
- Mobile-specific layouts and offline operation.
