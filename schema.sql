-- ============================================================================
-- Robot Arena Database Schema (robotarenainf)
-- Supports quiz tracking, sanity checks, and participant type differentiation
-- ============================================================================

-- Create the database (run this separately as superuser if needed)
-- CREATE DATABASE robotarenainf;

-- Connect to the database before running the rest:
-- \c robotarenainf

-- ============================================================================
-- Main Annotations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    
    -- Participant identification
    participant_id VARCHAR(255) NOT NULL,
    participant_type VARCHAR(50) NOT NULL DEFAULT 'unknown',  -- 'paid', 'free', or 'unknown'
    
    -- Completion tracking
    completion_code VARCHAR(50) NOT NULL,
    total_time_ms INTEGER NOT NULL,
    response_length INTEGER NOT NULL,
    
    -- Quiz tracking (for paid workers)
    quiz_score INTEGER,           -- Number of quiz questions answered correctly
    quiz_total INTEGER,           -- Total quiz questions shown
    
    -- Sanity check tracking (hidden from users)
    sanity_checks_passed INTEGER, -- Number of sanity checks passed
    sanity_checks_total INTEGER,  -- Total sanity checks encountered
    sanity_check_results JSONB,   -- Detailed results: [{index, correct, position, userAnswer, correctAnswer}]
    
    -- Failure tracking
    failed BOOLEAN DEFAULT FALSE,
    failure_reason VARCHAR(50),   -- 'quiz_failed' or 'sanity_failed'
    
    -- Response data
    response_data JSONB NOT NULL, -- Array of responses with type: 'quiz', 'sanity_check', or 'regular'
    
    -- Metadata
    timestamp TIMESTAMP NOT NULL,
    config_version VARCHAR(50),   -- e.g., 'v3.0'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indexes for better query performance
-- ============================================================================

-- Index for participant lookups
CREATE INDEX IF NOT EXISTS idx_participant_id ON annotations(participant_id);

-- Index for participant type filtering
CREATE INDEX IF NOT EXISTS idx_participant_type ON annotations(participant_type);

-- Index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_timestamp ON annotations(timestamp);

-- Index for failure tracking
CREATE INDEX IF NOT EXISTS idx_failed ON annotations(failed);
CREATE INDEX IF NOT EXISTS idx_failure_reason ON annotations(failure_reason);

-- Index for config version filtering
CREATE INDEX IF NOT EXISTS idx_config_version ON annotations(config_version);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_type_failed ON annotations(participant_type, failed);


-- ============================================================================
-- Useful Views
-- ============================================================================

-- View: Participant summary statistics
CREATE OR REPLACE VIEW participant_summary AS
SELECT 
    participant_type,
    COUNT(DISTINCT participant_id) AS total_participants,
    COUNT(DISTINCT CASE WHEN failed = false THEN participant_id END) AS completed,
    COUNT(DISTINCT CASE WHEN failure_reason = 'quiz_failed' THEN participant_id END) AS quiz_failed,
    COUNT(DISTINCT CASE WHEN failure_reason = 'sanity_failed' THEN participant_id END) AS sanity_failed,
    AVG(response_length) AS avg_responses,
    AVG(total_time_ms / 1000.0 / 60.0) AS avg_time_minutes
FROM annotations
GROUP BY participant_type;

-- View: Quiz performance for paid workers
CREATE OR REPLACE VIEW quiz_performance AS
SELECT 
    participant_id,
    quiz_score,
    quiz_total,
    ROUND(quiz_score::numeric / NULLIF(quiz_total, 0) * 100, 1) AS quiz_percentage,
    failed,
    failure_reason
FROM annotations
WHERE participant_type = 'paid' AND quiz_total IS NOT NULL;

-- View: Sanity check performance
CREATE OR REPLACE VIEW sanity_check_performance AS
SELECT 
    participant_id,
    participant_type,
    sanity_checks_passed,
    sanity_checks_total,
    ROUND(sanity_checks_passed::numeric / NULLIF(sanity_checks_total, 0) * 100, 1) AS sanity_percentage,
    failed,
    failure_reason
FROM annotations
WHERE sanity_checks_total IS NOT NULL AND sanity_checks_total > 0;


-- ============================================================================
-- Sample Queries
-- ============================================================================

-- Get all valid (non-failed) responses for ranking
-- SELECT response_data, timestamp, config_version 
-- FROM annotations 
-- WHERE failed = false 
-- ORDER BY timestamp ASC;

-- Get participant completion rates by type
-- SELECT * FROM participant_summary;

-- Get failed participants with reasons
-- SELECT participant_id, participant_type, failure_reason, timestamp 
-- FROM annotations 
-- WHERE failed = true 
-- ORDER BY timestamp DESC;

-- Count responses by type within response_data
-- SELECT 
--     jsonb_array_elements(response_data)->>'type' AS response_type,
--     COUNT(*) AS count
-- FROM annotations
-- GROUP BY response_type;

-- Get detailed sanity check failures
-- SELECT 
--     participant_id,
--     participant_type,
--     jsonb_array_elements(sanity_check_results) AS check_result
-- FROM annotations
-- WHERE failure_reason = 'sanity_failed';


-- ============================================================================
-- Migration from v2 (if needed)
-- ============================================================================

-- If you have existing data in the old 'annotations' table, you can migrate it:
-- 
-- INSERT INTO annotations (
--     participant_id, participant_type, completion_code, total_time_ms,
--     response_length, quiz_score, quiz_total, 
--     sanity_checks_passed, sanity_checks_total, sanity_check_results,
--     failed, failure_reason, response_data, timestamp, config_version
-- )
-- SELECT 
--     participant_id,
--     'unknown' AS participant_type,  -- Old data doesn't have this
--     completion_code,
--     total_time_ms,
--     response_length,
--     NULL AS quiz_score,
--     NULL AS quiz_total,
--     NULL AS sanity_checks_passed,
--     NULL AS sanity_checks_total,
--     '[]'::jsonb AS sanity_check_results,
--     FALSE AS failed,
--     NULL AS failure_reason,
--     response_data,
--     timestamp,
--     scaled_version AS config_version
-- FROM annotations;


-- ============================================================================
-- Permissions (adjust as needed)
-- ============================================================================

-- Grant permissions to your application user
-- GRANT SELECT, INSERT, UPDATE ON annotations TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE annotations_id_seq TO your_app_user;
-- GRANT SELECT ON participant_summary TO your_app_user;
-- GRANT SELECT ON quiz_performance TO your_app_user;
-- GRANT SELECT ON sanity_check_performance TO your_app_user;