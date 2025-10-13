// Worlds Tournament Simulator JavaScript

class HierarchicalBracket {
    constructor(eliminationResults) {
        // Validate input data structure
        this.validateEliminationResults(eliminationResults);
        
        this.eliminationResults = eliminationResults;
        this.container = null;
        this.isRendered = false;
    }

    validateEliminationResults(data) {
        // Data validation and error handling for malformed bracket data
        if (!data || typeof data !== 'object') {
            throw new Error('Elimination results must be a valid object');
        }

        // Check if quarterfinals data is present
        if (data.quarterfinals) {
            // Validate quarterfinals structure if present
            if (!Array.isArray(data.quarterfinals) || data.quarterfinals.length !== 4) {
                throw new Error('Quarterfinals must be an array of 4 matches');
            }

            data.quarterfinals.forEach((qf, index) => {
                if (!qf.match || !Array.isArray(qf.match) || qf.match.length !== 2) {
                    throw new Error(`Quarterfinal ${index + 1} must have exactly 2 teams`);
                }
                if (!qf.winner || !qf.match.includes(qf.winner)) {
                    throw new Error(`Quarterfinal ${index + 1} winner must be one of the match teams`);
                }
            });
        } else {
            // Generate placeholder quarterfinals from semifinals if missing
            console.warn('Quarterfinals data missing, generating placeholder data');
            data.quarterfinals = this.generatePlaceholderQuarterfinals(data.semifinals);
        }

        // Validate semifinals structure
        if (!Array.isArray(data.semifinals) || data.semifinals.length !== 2) {
            throw new Error('Semifinals must be an array of 2 matches');
        }

        data.semifinals.forEach((sf, index) => {
            if (!Array.isArray(sf) || sf.length !== 2) {
                throw new Error(`Semifinal ${index + 1} must have exactly 2 teams`);
            }
        });

        // Validate finals structure
        if (!Array.isArray(data.finals) || data.finals.length !== 2) {
            throw new Error('Finals must be an array of 2 teams');
        }

        // Validate champion
        if (!data.champion || typeof data.champion !== 'string') {
            throw new Error('Champion must be a valid team name');
        }

        if (!data.finals.includes(data.champion)) {
            throw new Error('Champion must be one of the finalists');
        }
    }

    generatePlaceholderQuarterfinals(semifinals) {
        // Generate placeholder quarterfinals data when missing
        const placeholderTeams = [
            'Team A', 'Team B', 'Team C', 'Team D', 
            'Team E', 'Team F', 'Team G', 'Team H'
        ];
        
        // Create quarterfinals that would lead to the given semifinals
        const qf1Winner = semifinals[0][0];
        const qf2Winner = semifinals[0][1];
        const qf3Winner = semifinals[1][0];
        const qf4Winner = semifinals[1][1];
        
        return [
            {
                match: [qf1Winner, placeholderTeams[0]],
                winner: qf1Winner
            },
            {
                match: [qf2Winner, placeholderTeams[1]],
                winner: qf2Winner
            },
            {
                match: [qf3Winner, placeholderTeams[2]],
                winner: qf3Winner
            },
            {
                match: [qf4Winner, placeholderTeams[3]],
                winner: qf4Winner
            }
        ];
    }

    renderBracket(containerSelector = '.elimination-bracket') {
        // Main method orchestrating all rendering
        try {
            this.container = document.querySelector(containerSelector);
            
            if (!this.container) {
                throw new Error(`Container element '${containerSelector}' not found`);
            }

            // Clear existing content
            this.container.innerHTML = '';

            // Create main bracket container
            const bracketContainer = document.createElement('div');
            bracketContainer.className = 'tournament-bracket-container';
            
            // Add accessibility attributes for screen readers
            bracketContainer.setAttribute('role', 'main');
            bracketContainer.setAttribute('aria-label', 'Tournament elimination bracket');
            bracketContainer.setAttribute('aria-describedby', 'bracket-description');
            bracketContainer.setAttribute('tabindex', '0');
            
            // Create hidden description for screen readers
            const bracketDescription = document.createElement('div');
            bracketDescription.id = 'bracket-description';
            bracketDescription.className = 'sr-only';
            
            // Generate comprehensive bracket description
            const qfTeams = this.eliminationResults.quarterfinals.map(qf => qf.match.join(' vs ')).join(', ');
            const sfTeams = this.eliminationResults.semifinals.map(sf => sf.join(' vs ')).join(', ');
            const finalists = this.eliminationResults.finals.join(' vs ');
            
            bracketDescription.textContent = `Tournament elimination bracket. Quarterfinals: ${qfTeams}. Semifinals: ${sfTeams}. Finals: ${finalists}. Champion: ${this.eliminationResults.champion}. Use arrow keys to navigate between rounds and matches. Press Enter or Space to hear match details.`;
            bracketContainer.appendChild(bracketDescription);

            // Render each stage in sequence
            const quarterfinalsElement = this.renderQuarterfinals();
            const qfToSfConnector = this.createConnector('qf-to-sf');
            const semifinalsElement = this.renderSemifinals();
            const sfToFConnector = this.createConnector('sf-to-f');
            const finalsElement = this.renderFinals();
            const championElement = this.renderChampion();

            // Append all elements to bracket container
            bracketContainer.appendChild(quarterfinalsElement);
            bracketContainer.appendChild(qfToSfConnector);
            bracketContainer.appendChild(semifinalsElement);
            bracketContainer.appendChild(sfToFConnector);
            bracketContainer.appendChild(finalsElement);
            bracketContainer.appendChild(championElement);

            // Add to main container
            this.container.appendChild(bracketContainer);

            // Mark as rendered and apply positioning
            this.isRendered = true;

            // Apply connection line positioning after DOM is updated
            setTimeout(() => {
                this.positionConnectionLines();
                // Initialize mobile interactions after rendering
                this.initializeMobileInteractions();
                // Initialize accessibility features after rendering
                this.initializeAccessibilityFeatures();
                // Enable keyboard navigation indicators
                bracketContainer.setAttribute('data-keyboard-navigation', 'true');
            }, 100);

            return bracketContainer;

        } catch (error) {
            console.error('Error rendering bracket:', error);
            this.renderErrorState(error.message);
            throw error;
        }
    }

    createConnector(type) {
        // Helper method to create connection elements
        const connector = document.createElement('div');
        connector.className = `bracket-connectors ${type}`;
        
        // Add accessibility attributes for connection lines
        connector.setAttribute('role', 'presentation');
        connector.setAttribute('aria-hidden', 'true');
        
        // Add descriptive text for screen readers (hidden)
        const description = document.createElement('span');
        description.className = 'sr-only';
        description.textContent = type === 'qf-to-sf' ? 
            'Connection lines from quarterfinals to semifinals' : 
            'Connection lines from semifinals to finals';
        connector.appendChild(description);
        
        return connector;
    }

    renderErrorState(errorMessage) {
        // Fallback error display
        if (this.container) {
            this.container.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <h5>Bracket Rendering Error</h5>
                    <p>${errorMessage}</p>
                    <small>Please check the elimination results data format.</small>
                </div>
            `;
        }
    }

    // Quarterfinals rendering logic
    renderQuarterfinals() {
        const quarterfinalsDiv = document.createElement('div');
        quarterfinalsDiv.className = 'bracket-round quarterfinals';
        quarterfinalsDiv.setAttribute('role', 'region');
        quarterfinalsDiv.setAttribute('aria-labelledby', 'qf-heading');
        
        // Add section heading for screen readers
        const heading = document.createElement('h3');
        heading.id = 'qf-heading';
        heading.className = 'sr-only';
        heading.textContent = 'Quarterfinals';
        quarterfinalsDiv.appendChild(heading);

        // Create top and bottom bracket halves
        const topHalf = document.createElement('div');
        topHalf.className = 'bracket-half top-half';
        
        const bottomHalf = document.createElement('div');
        bottomHalf.className = 'bracket-half bottom-half';

        // Position 4 matches in correct bracket halves (top/bottom structure)
        this.eliminationResults.quarterfinals.forEach((qf, index) => {
            const matchDiv = this.createMatchElement(qf, `QF ${index + 1}`, 'quarterfinal');
            
            // First two matches go to top half, last two to bottom half
            if (index < 2) {
                topHalf.appendChild(matchDiv);
            } else {
                bottomHalf.appendChild(matchDiv);
            }
        });

        quarterfinalsDiv.appendChild(topHalf);
        quarterfinalsDiv.appendChild(bottomHalf);

        return quarterfinalsDiv;
    }

    createMatchElement(matchData, matchLabel, roundType) {
        const matchDiv = document.createElement('div');
        matchDiv.className = 'bracket-match';
        matchDiv.setAttribute('data-round', roundType);
        matchDiv.setAttribute('data-match-id', matchLabel.toLowerCase().replace(' ', '-'));
        
        // Add accessibility attributes
        matchDiv.setAttribute('role', 'group');
        matchDiv.setAttribute('tabindex', '0');
        matchDiv.setAttribute('aria-label', `${matchLabel} match in ${roundType} round`);

        // Create match header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'bracket-match-header';
        headerDiv.textContent = matchLabel;

        // Create teams container
        const teamsDiv = document.createElement('div');
        teamsDiv.className = 'bracket-match-teams';

        // Handle different match data structures
        let team1, team2, winner, isPlaceholder = false;
        
        if (matchData.match && Array.isArray(matchData.match)) {
            // Quarterfinals structure: { match: [team1, team2], winner: team }
            team1 = matchData.match[0];
            team2 = matchData.match[1];
            winner = matchData.winner;
            
            // Check if this is a placeholder match (generated from missing data)
            isPlaceholder = team2.startsWith('Team ') && ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H'].includes(team2);
        } else if (Array.isArray(matchData)) {
            // Semifinals structure: [team1, team2]
            team1 = matchData[0];
            team2 = matchData[1];
            winner = null; // No winner determined yet for semifinals
        } else {
            throw new Error(`Invalid match data structure for ${matchLabel}`);
        }

        // Add placeholder styling if needed
        if (isPlaceholder) {
            matchDiv.classList.add('placeholder-match');
            headerDiv.textContent += ' (Reconstructed)';
        }

        // Create team slots with winner/loser visual states
        const team1Slot = this.createTeamSlot(team1, winner === team1 ? 'winner' : 'loser', isPlaceholder);
        const team2Slot = this.createTeamSlot(team2, winner === team2 ? 'winner' : 'loser', isPlaceholder);

        teamsDiv.appendChild(team1Slot);
        teamsDiv.appendChild(team2Slot);

        matchDiv.appendChild(headerDiv);
        matchDiv.appendChild(teamsDiv);
        
        // Add match result description for screen readers
        const matchResult = winner ? `Winner: ${winner}` : 'Match result pending';
        matchDiv.setAttribute('aria-describedby', `${matchLabel.toLowerCase().replace(' ', '-')}-result`);
        
        const resultDescription = document.createElement('div');
        resultDescription.id = `${matchLabel.toLowerCase().replace(' ', '-')}-result`;
        resultDescription.className = 'sr-only';
        resultDescription.textContent = `${matchLabel}: ${team1} vs ${team2}. ${matchResult}`;
        matchDiv.appendChild(resultDescription);

        return matchDiv;
    }

    createTeamSlot(teamName, state, isPlaceholder = false) {
        const teamSlot = document.createElement('div');
        teamSlot.className = `team-slot ${state}`;
        teamSlot.setAttribute('data-team', teamName);
        
        // Add placeholder styling if needed
        if (isPlaceholder && teamName.startsWith('Team ')) {
            teamSlot.classList.add('placeholder-team');
        }
        
        // Add accessibility attributes for team slots
        teamSlot.setAttribute('role', 'button');
        teamSlot.setAttribute('tabindex', '0');
        teamSlot.setAttribute('aria-label', `${teamName} - ${state === 'winner' ? 'Winner' : 'Eliminated'}`);
        teamSlot.setAttribute('aria-pressed', state === 'winner' ? 'true' : 'false');

        // Create team name display (short version)
        const teamNameDiv = document.createElement('div');
        teamNameDiv.className = 'team-name';
        teamNameDiv.textContent = this.getShortTeamName(teamName);

        // Create full team name for tooltip/hover
        const teamFullNameDiv = document.createElement('div');
        teamFullNameDiv.className = 'team-full-name';
        teamFullNameDiv.textContent = isPlaceholder && teamName.startsWith('Team ') ? 'Unknown Team' : teamName;

        teamSlot.appendChild(teamNameDiv);
        teamSlot.appendChild(teamFullNameDiv);

        return teamSlot;
    }

    getShortTeamName(teamName) {
        // Create short names for better display in bracket
        const shortNames = {
            'Gen.G eSports': 'GEN',
            'T1': 'T1',
            'Hanwha Life eSports': 'HLE',
            'KT Rolster': 'KT',
            'Bilibili Gaming': 'BLG',
            'Top Esports': 'TES',
            'Anyone s Legend': 'AL',
            'Invictus Gaming': 'IG',
            'G2 Esports': 'G2',
            'Fnatic': 'FNC',
            'Movistar KOI': 'KOI',
            'FlyQuest': 'FLY',
            '100 Thieves': '100T',
            'Vivo Keyd Stars': 'VKS',
            'PSG Talon': 'PSG',
            'CTBC Flying Oyster': 'CFO',
            'Team Secret Whales': 'TSW'
        };

        return shortNames[teamName] || teamName.substring(0, 3).toUpperCase();
    }

    renderSemifinals() {
        const semifinalsDiv = document.createElement('div');
        semifinalsDiv.className = 'bracket-round semifinals';
        semifinalsDiv.setAttribute('role', 'region');
        semifinalsDiv.setAttribute('aria-labelledby', 'sf-heading');
        
        // Add section heading for screen readers
        const heading = document.createElement('h3');
        heading.id = 'sf-heading';
        heading.className = 'sr-only';
        heading.textContent = 'Semifinals';
        semifinalsDiv.appendChild(heading);

        // Render SF matches aligning with QF winners
        this.eliminationResults.semifinals.forEach((teams, index) => {
            const matchDiv = this.createMatchElement(teams, `SF ${index + 1}`, 'semifinal');
            matchDiv.classList.add(`semifinal-${index + 1}`);
            semifinalsDiv.appendChild(matchDiv);
        });

        return semifinalsDiv;
    }

    renderFinals() {
        const finalsDiv = document.createElement('div');
        finalsDiv.className = 'bracket-round finals';
        finalsDiv.setAttribute('role', 'region');
        finalsDiv.setAttribute('aria-labelledby', 'finals-heading');
        
        // Add section heading for screen readers
        const heading = document.createElement('h3');
        heading.id = 'finals-heading';
        heading.className = 'sr-only';
        heading.textContent = 'Finals';
        finalsDiv.appendChild(heading);

        // Create final match centered between SF winners
        const finalMatchData = {
            match: this.eliminationResults.finals,
            winner: this.eliminationResults.champion
        };

        const finalMatchDiv = this.createMatchElement(finalMatchData, 'GRAND FINAL', 'final');
        finalsDiv.appendChild(finalMatchDiv);

        return finalsDiv;
    }

    renderChampion() {
        const championDiv = document.createElement('div');
        championDiv.className = 'champion-display';
        
        // Add accessibility attributes for champion display
        championDiv.setAttribute('role', 'banner');
        championDiv.setAttribute('aria-label', `Tournament Champion: ${this.eliminationResults.champion}`);
        championDiv.setAttribute('tabindex', '0');

        // Create champion display with special trophy styling
        const championIcon = document.createElement('div');
        championIcon.className = 'champion-icon';
        championIcon.innerHTML = '👑';
        championIcon.setAttribute('aria-hidden', 'true'); // Decorative icon

        const championTitle = document.createElement('div');
        championTitle.className = 'champion-title';
        championTitle.textContent = 'World Champion';

        const championTeam = document.createElement('div');
        championTeam.className = 'champion-team';
        championTeam.textContent = this.getShortTeamName(this.eliminationResults.champion);

        const championFullName = document.createElement('div');
        championFullName.className = 'team-full-name';
        championFullName.textContent = this.eliminationResults.champion;

        // Add trophy styling and animation
        championDiv.appendChild(championIcon);
        championDiv.appendChild(championTitle);
        championDiv.appendChild(championTeam);
        championDiv.appendChild(championFullName);

        // Add special champion styling
        championDiv.setAttribute('data-champion', this.eliminationResults.champion);

        return championDiv;
    }

    positionConnectionLines() {
        if (!this.isRendered || !this.container) {
            return;
        }

        // Calculate precise positioning for connection lines
        const qfMatches = this.container.querySelectorAll('.bracket-round.quarterfinals .bracket-match');
        const sfMatches = this.container.querySelectorAll('.bracket-round.semifinals .bracket-match');
        const finalMatch = this.container.querySelector('.bracket-round.finals .bracket-match');
        
        const qfToSfConnector = this.container.querySelector('.bracket-connectors.qf-to-sf');
        const sfToFConnector = this.container.querySelector('.bracket-connectors.sf-to-f');

        if (qfMatches.length >= 4 && sfMatches.length >= 2 && qfToSfConnector) {
            this.calculateQFToSFConnections(qfMatches, sfMatches, qfToSfConnector);
        }

        if (sfMatches.length >= 2 && finalMatch && sfToFConnector) {
            this.calculateSFToFinalConnections(sfMatches, finalMatch, sfToFConnector);
        }

        // Highlight winner path if tournament is complete
        if (this.eliminationResults.champion) {
            this.highlightWinnerPath();
        }
    }

    calculateQFToSFConnections(qfMatches, sfMatches, connector) {
        // Get positions of QF matches and SF matches
        const qfPositions = Array.from(qfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = this.container.getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                right: rect.right - containerRect.left,
                element: match
            };
        });

        const sfPositions = Array.from(sfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = this.container.getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                left: rect.left - containerRect.left,
                element: match
            };
        });

        // Calculate connection line positions
        if (qfPositions.length >= 4 && sfPositions.length >= 2) {
            // Calculate average positions for top and bottom bracket halves
            const topQFAvg = (qfPositions[0].top + qfPositions[1].top) / 2;
            const bottomQFAvg = (qfPositions[2].top + qfPositions[3].top) / 2;

            // Adjust connector height based on bracket spread
            const totalHeight = Math.abs(bottomQFAvg - topQFAvg) + 40;
            const centerY = (topQFAvg + bottomQFAvg) / 2;

            // Update CSS custom properties for dynamic positioning
            connector.style.setProperty('--connector-height', `${totalHeight}px`);
            connector.style.setProperty('--connector-top', `${centerY - totalHeight/2}px`);
        }
    }

    calculateSFToFinalConnections(sfMatches, finalMatch, connector) {
        // Get positions of SF matches and Final match
        const sfPositions = Array.from(sfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = this.container.getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                right: rect.right - containerRect.left
            };
        });

        const finalRect = finalMatch.getBoundingClientRect();
        const containerRect = this.container.getBoundingClientRect();
        const finalPosition = {
            top: finalRect.top - containerRect.top + finalRect.height / 2,
            left: finalRect.left - containerRect.left
        };

        if (sfPositions.length >= 2) {
            // Calculate connection positioning for SF to Final
            const sf1Top = sfPositions[0].top;
            const sf2Top = sfPositions[1].top;

            const totalHeight = Math.abs(sf2Top - sf1Top) + 20;
            const centerY = (sf1Top + sf2Top) / 2;

            // Update CSS custom properties for SF to Final connections
            connector.style.setProperty('--connector-height', `${totalHeight}px`);
            connector.style.setProperty('--connector-top', `${centerY - totalHeight/2}px`);
        }
    }

    highlightWinnerPath() {
        // Trace champion's complete route from QF to champion
        if (!this.eliminationResults.champion || !this.container) {
            return;
        }

        const champion = this.eliminationResults.champion;
        
        // Find champion's quarterfinal match
        const championQF = this.eliminationResults.quarterfinals.find(qf => qf.winner === champion);
        if (!championQF) {
            console.warn('Champion quarterfinal match not found');
            return;
        }

        // Find champion's semifinal match
        const championSF = this.eliminationResults.semifinals.find(sf => sf.includes(champion));
        if (!championSF) {
            console.warn('Champion semifinal match not found');
            return;
        }

        // Highlight champion's path through each round
        this.highlightChampionMatches(champion);
        this.highlightConnectionLines();
        this.addPathAnimations();
    }

    highlightChampionMatches(champion) {
        // Add visual emphasis to all matches in champion's path
        const allMatches = this.container.querySelectorAll('.bracket-match');
        
        allMatches.forEach(match => {
            const winnerSlots = match.querySelectorAll('.team-slot.winner');
            const hasChampion = Array.from(winnerSlots).some(slot => 
                slot.getAttribute('data-team') === champion
            );
            
            if (hasChampion) {
                match.classList.add('champion-path-match');
                
                // Add special styling to champion's team slot
                winnerSlots.forEach(slot => {
                    if (slot.getAttribute('data-team') === champion) {
                        slot.classList.add('champion-team-slot');
                    }
                });
            }
        });

        // Special highlighting for champion display
        const championDisplay = this.container.querySelector('.champion-display');
        if (championDisplay) {
            championDisplay.classList.add('champion-celebration');
        }
    }

    highlightConnectionLines() {
        // Add visual emphasis for the complete tournament path
        const qfToSfConnector = this.container.querySelector('.bracket-connectors.qf-to-sf');
        const sfToFConnector = this.container.querySelector('.bracket-connectors.sf-to-f');

        // Add active path class to highlight winner's route
        if (qfToSfConnector) {
            qfToSfConnector.classList.add('active-path', 'champion-path');
        }
        if (sfToFConnector) {
            sfToFConnector.classList.add('active-path', 'champion-path');
        }
    }

    addPathAnimations() {
        // Create smooth transition animations for path highlighting
        const pathElements = this.container.querySelectorAll('.champion-path-match, .champion-path, .champion-celebration');
        
        // Apply staggered animation timing
        pathElements.forEach((element, index) => {
            element.style.transition = 'all 0.8s ease-in-out';
            element.style.animationDelay = `${index * 0.2}s`;
            
            // Add pulsing animation for champion path
            setTimeout(() => {
                element.classList.add('path-animated');
            }, index * 200);
        });

        // Special animation for champion celebration
        const championDisplay = this.container.querySelector('.champion-celebration');
        if (championDisplay) {
            setTimeout(() => {
                championDisplay.classList.add('champion-pulse');
            }, pathElements.length * 200);
        }
    }



    // Mobile interaction enhancements
    initializeMobileInteractions() {
        if (!this.isMobileDevice()) {
            return;
        }

        // Add touch-friendly team name truncation with tap-to-expand
        this.setupTapToExpandTeamNames();
        
        // Add swipe gestures for bracket navigation
        this.setupSwipeGestures();
        
        // Add mobile-optimized connection indicators
        this.addMobileConnectionIndicators();
        
        // Add swipe indicators
        this.addSwipeIndicators();
    }

    isMobileDevice() {
        return window.innerWidth <= 768 || 
               /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    setupTapToExpandTeamNames() {
        const teamSlots = this.container.querySelectorAll('.team-slot');
        
        teamSlots.forEach(slot => {
            // Add touch event listeners for tap-to-expand functionality
            slot.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.handleTeamSlotTap(slot);
            });
            
            slot.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleTeamSlotTap(slot);
            });
        });

        // Close expanded team slots when tapping elsewhere
        document.addEventListener('touchstart', (e) => {
            if (!e.target.closest('.team-slot')) {
                this.closeAllExpandedTeamSlots();
            }
        });
    }

    handleTeamSlotTap(slot) {
        const isExpanded = slot.classList.contains('expanded');
        
        // Close all other expanded slots
        this.closeAllExpandedTeamSlots();
        
        if (!isExpanded) {
            // Expand this slot
            slot.classList.add('expanded');
            
            // Announce expansion for screen readers
            if (this.liveRegion) {
                const teamName = slot.getAttribute('data-team');
                const fullName = slot.querySelector('.team-full-name')?.textContent || teamName;
                this.liveRegion.textContent = `Expanded team details for ${fullName}`;
            }
            
            // Auto-close after 3 seconds
            setTimeout(() => {
                slot.classList.remove('expanded');
            }, 3000);
        }
    }

    closeAllExpandedTeamSlots() {
        const expandedSlots = this.container.querySelectorAll('.team-slot.expanded');
        expandedSlots.forEach(slot => {
            slot.classList.remove('expanded');
        });
    }

    setupSwipeGestures() {
        let startX = 0;
        let startY = 0;
        let isScrolling = false;

        this.container.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
            isScrolling = false;
        });

        this.container.addEventListener('touchmove', (e) => {
            if (!startX || !startY) {
                return;
            }

            const currentX = e.touches[0].clientX;
            const currentY = e.touches[0].clientY;
            
            const diffX = Math.abs(currentX - startX);
            const diffY = Math.abs(currentY - startY);

            // Determine if this is a horizontal swipe
            if (diffX > diffY && diffX > 30) {
                isScrolling = true;
                
                // Enhance native scrolling with momentum
                if (currentX < startX) {
                    // Swiping left - scroll right
                    this.smoothScrollTo(this.container.scrollLeft + 100);
                } else {
                    // Swiping right - scroll left
                    this.smoothScrollTo(this.container.scrollLeft - 100);
                }
            }
        });

        this.container.addEventListener('touchend', () => {
            startX = 0;
            startY = 0;
            isScrolling = false;
        });
    }

    smoothScrollTo(targetScrollLeft) {
        const container = this.container;
        const startScrollLeft = container.scrollLeft;
        const distance = targetScrollLeft - startScrollLeft;
        const duration = 300;
        let startTime = null;

        function animation(currentTime) {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const progress = Math.min(timeElapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeInOutCubic = progress < 0.5 
                ? 4 * progress * progress * progress 
                : (progress - 1) * (2 * progress - 2) * (2 * progress - 2) + 1;
            
            container.scrollLeft = startScrollLeft + distance * easeInOutCubic;
            
            if (progress < 1) {
                requestAnimationFrame(animation);
            }
        }

        requestAnimationFrame(animation);
    }

    addMobileConnectionIndicators() {
        const bracketRounds = this.container.querySelectorAll('.bracket-round:not(.champion-display)');
        
        bracketRounds.forEach((round, index) => {
            if (index < bracketRounds.length - 1) {
                const indicator = document.createElement('div');
                indicator.className = 'mobile-connection-indicator';
                round.style.position = 'relative';
                round.appendChild(indicator);
            }
        });
    }

    addSwipeIndicators() {
        // Add swipe indicators for better UX
        const leftIndicator = document.createElement('div');
        leftIndicator.className = 'swipe-indicator left';
        leftIndicator.innerHTML = '←';
        
        const rightIndicator = document.createElement('div');
        rightIndicator.className = 'swipe-indicator right';
        rightIndicator.innerHTML = '→';
        
        this.container.style.position = 'relative';
        this.container.appendChild(leftIndicator);
        this.container.appendChild(rightIndicator);
        
        // Hide indicators after 5 seconds
        setTimeout(() => {
            leftIndicator.style.opacity = '0';
            rightIndicator.style.opacity = '0';
            setTimeout(() => {
                leftIndicator.remove();
                rightIndicator.remove();
            }, 500);
        }, 5000);
        
        // Show/hide indicators based on scroll position
        this.container.addEventListener('scroll', () => {
            const scrollLeft = this.container.scrollLeft;
            const maxScroll = this.container.scrollWidth - this.container.clientWidth;
            
            leftIndicator.style.opacity = scrollLeft > 0 ? '0.3' : '0';
            rightIndicator.style.opacity = scrollLeft < maxScroll ? '0.3' : '0';
        });
    }

    // Accessibility Features
    initializeAccessibilityFeatures() {
        if (!this.container) {
            return;
        }

        // Initialize keyboard navigation
        this.setupKeyboardNavigation();
        
        // Add focus management
        this.setupFocusManagement();
        
        // Add live region for dynamic updates
        this.setupLiveRegion();
        
        // Add skip links for better navigation
        this.addSkipLinks();
    }

    setupKeyboardNavigation() {
        const bracketContainer = this.container.querySelector('.tournament-bracket-container');
        if (!bracketContainer) return;

        // Get all focusable elements in bracket
        const focusableElements = bracketContainer.querySelectorAll('[tabindex="0"]');
        let currentFocusIndex = 0;

        bracketContainer.addEventListener('keydown', (e) => {
            switch (e.key) {
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigateToNextRound(focusableElements, currentFocusIndex);
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigateToPreviousRound(focusableElements, currentFocusIndex);
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    this.navigateToNextMatch(focusableElements, currentFocusIndex);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.navigateToPreviousMatch(focusableElements, currentFocusIndex);
                    break;
                case 'Enter':
                case ' ':
                    e.preventDefault();
                    this.activateCurrentElement(e.target);
                    break;
                case 'Home':
                    e.preventDefault();
                    this.focusFirstElement(focusableElements);
                    break;
                case 'End':
                    e.preventDefault();
                    this.focusLastElement(focusableElements);
                    break;
            }
        });

        // Track current focus for navigation
        focusableElements.forEach((element, index) => {
            element.addEventListener('focus', () => {
                currentFocusIndex = index;
                this.updateLiveRegion(element);
            });
        });
    }

    navigateToNextRound(elements, currentIndex) {
        const currentElement = elements[currentIndex];
        const currentRound = currentElement.getAttribute('data-round') || 
                           currentElement.closest('[data-round]')?.getAttribute('data-round');
        
        // Find next round element
        const nextRoundElement = this.findElementInNextRound(elements, currentRound, currentIndex);
        if (nextRoundElement) {
            nextRoundElement.focus();
        }
    }

    navigateToPreviousRound(elements, currentIndex) {
        const currentElement = elements[currentIndex];
        const currentRound = currentElement.getAttribute('data-round') || 
                           currentElement.closest('[data-round]')?.getAttribute('data-round');
        
        // Find previous round element
        const prevRoundElement = this.findElementInPreviousRound(elements, currentRound, currentIndex);
        if (prevRoundElement) {
            prevRoundElement.focus();
        }
    }

    navigateToNextMatch(elements, currentIndex) {
        const nextIndex = Math.min(currentIndex + 1, elements.length - 1);
        if (nextIndex !== currentIndex) {
            elements[nextIndex].focus();
        }
    }

    navigateToPreviousMatch(elements, currentIndex) {
        const prevIndex = Math.max(currentIndex - 1, 0);
        if (prevIndex !== currentIndex) {
            elements[prevIndex].focus();
        }
    }

    findElementInNextRound(elements, currentRound, currentIndex) {
        const roundOrder = ['quarterfinal', 'semifinal', 'final', 'champion'];
        const currentRoundIndex = roundOrder.indexOf(currentRound);
        
        if (currentRoundIndex < roundOrder.length - 1) {
            const nextRound = roundOrder[currentRoundIndex + 1];
            
            for (let i = currentIndex + 1; i < elements.length; i++) {
                const element = elements[i];
                const elementRound = element.getAttribute('data-round') || 
                                  element.closest('[data-round]')?.getAttribute('data-round') ||
                                  (element.classList.contains('champion-display') ? 'champion' : null);
                
                if (elementRound === nextRound) {
                    return element;
                }
            }
        }
        return null;
    }

    findElementInPreviousRound(elements, currentRound, currentIndex) {
        const roundOrder = ['quarterfinal', 'semifinal', 'final', 'champion'];
        const currentRoundIndex = roundOrder.indexOf(currentRound);
        
        if (currentRoundIndex > 0) {
            const prevRound = roundOrder[currentRoundIndex - 1];
            
            for (let i = currentIndex - 1; i >= 0; i--) {
                const element = elements[i];
                const elementRound = element.getAttribute('data-round') || 
                                  element.closest('[data-round]')?.getAttribute('data-round');
                
                if (elementRound === prevRound) {
                    return element;
                }
            }
        }
        return null;
    }

    activateCurrentElement(element) {
        // Handle activation of focused element
        if (element.classList.contains('team-slot')) {
            this.announceTeamDetails(element);
        } else if (element.classList.contains('bracket-match')) {
            this.announceMatchDetails(element);
        } else if (element.classList.contains('champion-display')) {
            this.announceChampionDetails(element);
        }
    }

    focusFirstElement(elements) {
        if (elements.length > 0) {
            elements[0].focus();
        }
    }

    focusLastElement(elements) {
        if (elements.length > 0) {
            elements[elements.length - 1].focus();
        }
    }

    setupFocusManagement() {
        const bracketContainer = this.container.querySelector('.tournament-bracket-container');
        if (!bracketContainer) return;

        // Add focus indicators
        const style = document.createElement('style');
        style.textContent = `
            .tournament-bracket-container [tabindex="0"]:focus {
                outline: 3px solid #007bff;
                outline-offset: 2px;
                box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
            }
            
            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }
        `;
        document.head.appendChild(style);
    }

    setupLiveRegion() {
        // Create live region for announcing dynamic changes
        const liveRegion = document.createElement('div');
        liveRegion.id = 'bracket-live-region';
        liveRegion.className = 'sr-only';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        
        this.container.appendChild(liveRegion);
        this.liveRegion = liveRegion;
    }

    updateLiveRegion(element) {
        if (!this.liveRegion) return;

        let announcement = '';
        
        if (element.classList.contains('bracket-match')) {
            const matchLabel = element.querySelector('.bracket-match-header')?.textContent || 'Match';
            const teams = Array.from(element.querySelectorAll('.team-slot')).map(slot => 
                slot.getAttribute('data-team')).join(' vs ');
            announcement = `Focused on ${matchLabel}: ${teams}`;
        } else if (element.classList.contains('team-slot')) {
            const teamName = element.getAttribute('data-team');
            const state = element.classList.contains('winner') ? 'Winner' : 'Eliminated';
            announcement = `Focused on ${teamName}, ${state}`;
        } else if (element.classList.contains('champion-display')) {
            const champion = element.getAttribute('data-champion');
            announcement = `Focused on Tournament Champion: ${champion}`;
        }

        if (announcement) {
            this.liveRegion.textContent = announcement;
        }
    }

    addSkipLinks() {
        const bracketContainer = this.container.querySelector('.tournament-bracket-container');
        if (!bracketContainer) return;

        // Add skip to rounds links and accessibility instructions
        const skipLinksContainer = document.createElement('div');
        skipLinksContainer.className = 'skip-links sr-only';
        skipLinksContainer.innerHTML = `
            <div class="accessibility-instructions">
                <h2>Bracket Navigation Instructions</h2>
                <p>Use arrow keys to navigate: Right/Left for rounds, Up/Down for matches. Press Enter or Space for details. Home/End to jump to first/last element.</p>
            </div>
            <a href="#quarterfinals-section" class="skip-link">Skip to Quarterfinals</a>
            <a href="#semifinals-section" class="skip-link">Skip to Semifinals</a>
            <a href="#finals-section" class="skip-link">Skip to Finals</a>
            <a href="#champion-section" class="skip-link">Skip to Champion</a>
        `;

        // Add IDs to sections for skip links
        const quarterfinalsSection = bracketContainer.querySelector('.bracket-round.quarterfinals');
        const semifinalsSection = bracketContainer.querySelector('.bracket-round.semifinals');
        const finalsSection = bracketContainer.querySelector('.bracket-round.finals');
        const championSection = bracketContainer.querySelector('.champion-display');

        if (quarterfinalsSection) quarterfinalsSection.id = 'quarterfinals-section';
        if (semifinalsSection) semifinalsSection.id = 'semifinals-section';
        if (finalsSection) finalsSection.id = 'finals-section';
        if (championSection) championSection.id = 'champion-section';

        bracketContainer.insertBefore(skipLinksContainer, bracketContainer.firstChild);

        // Style skip links to be visible on focus
        const skipLinkStyle = document.createElement('style');
        skipLinkStyle.textContent = `
            .skip-link:focus {
                position: absolute;
                top: 10px;
                left: 10px;
                background: #007bff;
                color: white;
                padding: 8px 12px;
                text-decoration: none;
                border-radius: 4px;
                z-index: 1000;
                clip: auto;
                width: auto;
                height: auto;
            }
        `;
        document.head.appendChild(skipLinkStyle);
    }

    announceTeamDetails(teamSlot) {
        const teamName = teamSlot.getAttribute('data-team');
        const isWinner = teamSlot.classList.contains('winner');
        const fullName = teamSlot.querySelector('.team-full-name')?.textContent || teamName;
        
        const announcement = `${fullName}, ${isWinner ? 'Advanced to next round' : 'Eliminated from tournament'}`;
        
        if (this.liveRegion) {
            this.liveRegion.textContent = announcement;
        }
    }

    announceMatchDetails(matchElement) {
        const matchLabel = matchElement.querySelector('.bracket-match-header')?.textContent || 'Match';
        const teams = Array.from(matchElement.querySelectorAll('.team-slot'));
        const team1 = teams[0]?.getAttribute('data-team') || 'Team 1';
        const team2 = teams[1]?.getAttribute('data-team') || 'Team 2';
        const winner = teams.find(team => team.classList.contains('winner'))?.getAttribute('data-team');
        
        let announcement = `${matchLabel}: ${team1} versus ${team2}`;
        if (winner) {
            announcement += `. Winner: ${winner}`;
        }
        
        if (this.liveRegion) {
            this.liveRegion.textContent = announcement;
        }
    }

    announceChampionDetails(championElement) {
        const champion = championElement.getAttribute('data-champion');
        const announcement = `Tournament Champion: ${champion}. Congratulations to the World Champions!`;
        
        if (this.liveRegion) {
            this.liveRegion.textContent = announcement;
        }
    }
}

class TournamentSimulator {
    constructor() {
        this.initializeEventListeners();
        this.loadRegionalStrengths();
        this.loadTeamRankings();
    }

    initializeEventListeners() {
        // Simulation type change
        document.getElementById('simulationType').addEventListener('change', (e) => {
            const numSimDiv = document.getElementById('numSimulationsDiv');
            if (e.target.value === 'multiple') {
                numSimDiv.style.display = 'block';
            } else {
                numSimDiv.style.display = 'none';
            }
        });

        // Run simulation button
        document.getElementById('runSimulation').addEventListener('click', () => {
            this.runSimulation();
        });

        // Window resize for responsive connection adjustments
        window.addEventListener('resize', () => {
            this.adjustConnectionsForScreenSize();
            // Recalculate connection positions on resize
            setTimeout(() => {
                const eliminationBracket = document.querySelector('.tournament-bracket-container');
                if (eliminationBracket) {
                    // Re-trigger positioning if bracket exists
                    const qfMatches = document.querySelectorAll('.bracket-round.quarterfinals .bracket-match');
                    const sfMatches = document.querySelectorAll('.bracket-round.semifinals .bracket-match');
                    const finalMatch = document.querySelector('.bracket-round.finals .bracket-match');
                    
                    if (qfMatches.length > 0 && sfMatches.length > 0 && finalMatch) {
                        this.positionConnectionLines({
                            quarterfinals: [],
                            semifinals: [],
                            finals: [],
                            champion: null
                        });
                    }
                }
            }, 100);
        });
    }

    async loadRegionalStrengths() {
        try {
            const response = await fetch('/api/regional_strengths');
            const data = await response.json();

            if (data.success) {
                this.displayRegionalStrengths(data.regional_strengths);
            }
        } catch (error) {
            console.error('Error loading regional strengths:', error);
        }
    }

    displayRegionalStrengths(strengths) {
        const container = document.getElementById('regionalStrengths');
        container.innerHTML = '';

        // Sort by strength
        const sortedRegions = Object.entries(strengths)
            .sort(([, a], [, b]) => b.strength - a.strength);

        sortedRegions.forEach(([region, data]) => {
            const item = document.createElement('div');
            item.className = 'regional-strength-item';

            const percentage = (data.strength * 100).toFixed(1);

            item.innerHTML = `
                <div>
                    <strong>${region}</strong>
                    <br>
                    <small class="text-muted">${data.description}</small>
                </div>
                <div class="text-end">
                    <div class="strength-bar">
                        <div class="strength-fill region-${region.toLowerCase()}" 
                             style="width: ${percentage}%"></div>
                    </div>
                    <small>${percentage}%</small>
                </div>
            `;

            container.appendChild(item);
        });
    }

    async runSimulation() {
        const simulationType = document.getElementById('simulationType').value;
        const numSimulations = document.getElementById('numSimulations').value;

        // Show loading
        this.showLoading(true);

        try {
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: simulationType,
                    num_simulations: parseInt(numSimulations)
                })
            });

            const data = await response.json();

            if (data.success) {
                if (data.type === 'single') {
                    this.displaySingleResults(data.result);
                } else {
                    this.displayMultipleResults(data);
                }
            } else {
                this.showError(data.error);
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displaySingleResults(result) {
        // Hide welcome message and multiple results
        document.getElementById('resultsArea').style.display = 'none';
        document.getElementById('multipleResults').style.display = 'none';

        // Show single results
        document.getElementById('singleResults').style.display = 'block';

        // Populate champion info
        document.getElementById('championName').textContent = result.champion;
        document.getElementById('championRegion').textContent = `${result.champion_region} Region`;

        // Populate play-in results
        document.getElementById('playinWinner').textContent = result.playin_winner;
        document.getElementById('playinLoser').textContent = result.playin_loser;

        // Populate Swiss results
        const qualifiedList = document.getElementById('swissQualified');
        const eliminatedList = document.getElementById('swissEliminated');

        qualifiedList.innerHTML = '';
        eliminatedList.innerHTML = '';

        result.swiss_qualified.forEach(team => {
            const li = document.createElement('li');
            li.className = 'team-list-item qualified-item';
            const record = result.swiss_records[team];
            li.innerHTML = `<strong>${team}</strong> <small>(${record[0]}-${record[1]})</small>`;
            qualifiedList.appendChild(li);
        });

        result.swiss_eliminated.forEach(team => {
            const li = document.createElement('li');
            li.className = 'team-list-item eliminated-item';
            const record = result.swiss_records[team];
            li.innerHTML = `<strong>${team}</strong> <small>(${record[0]}-${record[1]})</small>`;
            eliminatedList.appendChild(li);
        });



        // Display elimination bracket
        this.displayEliminationBracket(result.elimination_results);
    }

    async displayMultipleResults(data) {
        // Hide welcome message and single results
        document.getElementById('resultsArea').style.display = 'none';
        document.getElementById('singleResults').style.display = 'none';

        // Show multiple results
        document.getElementById('multipleResults').style.display = 'block';

        // Populate championship stats table
        const tableBody = document.getElementById('championStatsTable');
        tableBody.innerHTML = '';

        data.champion_stats.slice(0, 10).forEach((stat, index) => {
            const row = document.createElement('tr');
            const [team, wins, probability] = stat;

            row.innerHTML = `
                <td><strong>#${index + 1}</strong></td>
                <td>${team}</td>
                <td>${wins}</td>
                <td>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${probability * 100}%"
                             aria-valuenow="${probability * 100}" 
                             aria-valuemin="0" aria-valuemax="100">
                            ${(probability * 100).toFixed(1)}%
                        </div>
                    </div>
                </td>
            `;

            tableBody.appendChild(row);
        });

        // Display regional distribution
        this.displayRegionalDistribution(data.regional_distribution, data.num_simulations);

        // Generate and display chart
        await this.generateChart(data.champion_stats);
    }

    displayRegionalDistribution(regionWins, totalSims) {
        const container = document.getElementById('regionalDistribution');
        container.innerHTML = '';

        // Sort by wins
        const sortedRegions = Object.entries(regionWins)
            .sort(([, a], [, b]) => b - a);

        sortedRegions.forEach(([region, wins]) => {
            const percentage = (wins / totalSims * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = 'mb-2';
            item.innerHTML = `
                <div class="d-flex justify-content-between">
                    <strong>${region}</strong>
                    <span>${wins} (${percentage}%)</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar region-${region.toLowerCase()}" 
                         style="width: ${percentage}%"></div>
                </div>
            `;

            container.appendChild(item);
        });
    }

    async generateChart(championStats) {
        try {
            const response = await fetch(`/api/generate_chart?data=${encodeURIComponent(JSON.stringify(championStats))}`);
            const data = await response.json();

            if (data.success) {
                const chartContainer = document.getElementById('championshipChart');
                chartContainer.innerHTML = `<img src="${data.chart}" class="img-fluid" alt="Championship Probabilities Chart">`;
            }
        } catch (error) {
            console.error('Error generating chart:', error);
        }
    }

    displaySwissStageDetails(swissHistory) {
        const container = document.getElementById('swissStageDetails');
        container.innerHTML = '';

        // Create grid-based tournament layout like worldssim.com
        const gridContainer = document.createElement('div');
        gridContainer.className = 'swiss-grid-container';

        // Group matches by round and organize by records
        const roundsData = [];

        swissHistory.forEach((round, roundIndex) => {
            const roundData = {
                roundNumber: roundIndex + 1,
                roundName: round.round_name,
                matches: round.matches
            };
            roundsData.push(roundData);
        });

        // Create columns for each round
        roundsData.forEach((round, roundIndex) => {
            const roundColumn = document.createElement('div');
            roundColumn.className = 'swiss-round-column';

            // Round header with teal background
            const roundHeader = document.createElement('div');
            roundHeader.className = 'swiss-round-header';
            roundHeader.textContent = `Round ${round.roundNumber}`;
            roundColumn.appendChild(roundHeader);

            // Matches grid
            const matchesGrid = document.createElement('div');
            matchesGrid.className = 'round-matches-grid';

            round.matches.forEach((match, matchIndex) => {
                const matchCard = document.createElement('div');
                matchCard.className = 'swiss-match-card';

                const team1Short = this.getShortTeamName(match.team1);
                const team2Short = this.getShortTeamName(match.team2);

                matchCard.innerHTML = `
                    <div class="match-teams-grid">
                        <div class="team-box ${match.winner === match.team1 ? 'winner' : ''}">
                            <div class="team-logo-grid">${team1Short}</div>
                            <div class="team-name-grid">${match.team1}</div>
                        </div>
                        <div class="vs-text-grid">VS</div>
                        <div class="team-box ${match.winner === match.team2 ? 'winner' : ''}">
                            <div class="team-logo-grid">${team2Short}</div>
                            <div class="team-name-grid">${match.team2}</div>
                        </div>
                    </div>
                    ${match.best_of > 1 ? '<div class="elimination-indicator">Elimination Match</div>' : ''}
                `;

                matchesGrid.appendChild(matchCard);
            });

            roundColumn.appendChild(matchesGrid);
            gridContainer.appendChild(roundColumn);
        });

        container.appendChild(gridContainer);
    }

    getShortTeamName(teamName) {
        // Create short names for better display
        const shortNames = {
            'Gen.G eSports': 'GEN',
            'T1': 'T1',
            'Hanwha Life eSports': 'HLE',
            'KT Rolster': 'KT',
            'Bilibili Gaming': 'BLG',
            'Top Esports': 'TES',
            'Anyone s Legend': 'AL',
            'Invictus Gaming': 'IG',
            'G2 Esports': 'G2',
            'Fnatic': 'FNC',
            'Movistar KOI': 'KOI',
            'FlyQuest': 'FLY',
            '100 Thieves': '100T',
            'Vivo Keyd Stars': 'VKS',
            'PSG Talon': 'PSG',
            'CTBC Flying Oyster': 'CFO',
            'Team Secret Whales': 'TSW'
        };

        return shortNames[teamName] || teamName.substring(0, 3).toUpperCase();
    }

    displayEliminationBracket(eliminationResults) {
        // Try to render the visual bracket, fall back to simple display if it fails
        try {
            console.log('Rendering elimination bracket with data:', eliminationResults);
            
            // First try the simple visual bracket
            this.renderSimpleBracket(eliminationResults);
            
        } catch (error) {
            console.error('Error rendering bracket:', error);
            // Fallback to basic text display
            this.displayBasicEliminationBracket(eliminationResults);
        }
    }

    renderSimpleBracket(eliminationResults) {
        const container = document.querySelector('.elimination-bracket');
        
        // Create a simple visual bracket using flexbox instead of complex grid
        container.innerHTML = `
            <div class="simple-bracket-container">
                <div class="bracket-stage">
                    <h5 class="stage-title">Quarterfinals</h5>
                    <div class="stage-matches">
                        ${eliminationResults.quarterfinals ? eliminationResults.quarterfinals.map((qf, index) => `
                            <div class="simple-match">
                                <div class="match-header">QF ${index + 1}</div>
                                <div class="match-teams">
                                    <div class="team ${qf.winner === qf.match[0] ? 'winner' : 'loser'}">${qf.match[0]}</div>
                                    <div class="team ${qf.winner === qf.match[1] ? 'winner' : 'loser'}">${qf.match[1]}</div>
                                </div>
                                <div class="match-result">Winner: ${qf.winner}</div>
                            </div>
                        `).join('') : '<p class="text-muted">Quarterfinals data not available</p>'}
                    </div>
                </div>
                
                <div class="bracket-stage">
                    <h5 class="stage-title">Semifinals</h5>
                    <div class="stage-matches">
                        ${eliminationResults.semifinals.map((sf, index) => `
                            <div class="simple-match">
                                <div class="match-header">SF ${index + 1}</div>
                                <div class="match-teams">
                                    <div class="team">${sf[0]}</div>
                                    <div class="team">${sf[1]}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="bracket-stage">
                    <h5 class="stage-title">Finals</h5>
                    <div class="stage-matches">
                        <div class="simple-match final-match">
                            <div class="match-header">GRAND FINAL</div>
                            <div class="match-teams">
                                <div class="team ${eliminationResults.champion === eliminationResults.finals[0] ? 'winner' : 'loser'}">${eliminationResults.finals[0]}</div>
                                <div class="team ${eliminationResults.champion === eliminationResults.finals[1] ? 'winner' : 'loser'}">${eliminationResults.finals[1]}</div>
                            </div>
                            <div class="match-result">Winner: ${eliminationResults.champion}</div>
                        </div>
                    </div>
                </div>
                
                <div class="bracket-stage">
                    <div class="champion-display-simple">
                        <div class="champion-icon">🏆</div>
                        <div class="champion-title">World Champion</div>
                        <div class="champion-name">${eliminationResults.champion}</div>
                    </div>
                </div>
            </div>
        `;
    }

    displayBasicEliminationBracket(eliminationResults) {
        // Fallback basic bracket display for error cases
        const container = document.querySelector('.elimination-bracket');
        
        let quarterfinalsHtml = '';
        if (eliminationResults.quarterfinals && eliminationResults.quarterfinals.length > 0) {
            quarterfinalsHtml = `
                <div class="mb-3">
                    <h6>Quarterfinals:</h6>
                    <ul class="list-unstyled">
                        ${eliminationResults.quarterfinals.map(qf => 
                            `<li><strong>${qf.winner}</strong> def. ${qf.match.find(team => team !== qf.winner)}</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }
        
        container.innerHTML = `
            <div class="alert alert-info" role="alert">
                <h5><i class="fas fa-sitemap"></i> Elimination Bracket</h5>
                ${quarterfinalsHtml}
                <div class="mb-3">
                    <h6>Semifinals:</h6>
                    <ul class="list-unstyled">
                        ${eliminationResults.semifinals.map(sf => `<li>${sf.join(' vs ')}</li>`).join('')}
                    </ul>
                </div>
                <div class="mb-3">
                    <h6>Finals:</h6>
                    <p>${eliminationResults.finals.join(' vs ')}</p>
                </div>
                <div class="champion-highlight">
                    <h4><i class="fas fa-crown"></i> Champion: ${eliminationResults.champion}</h4>
                </div>
                <small class="text-muted">Visual bracket display temporarily unavailable. Running new simulation will show full bracket.</small>
            </div>
        `;
    }



    positionConnectionLines(eliminationResults) {
        // Calculate precise positioning for connection lines
        const qfMatches = document.querySelectorAll('.bracket-round.quarterfinals .bracket-match');
        const sfMatches = document.querySelectorAll('.bracket-round.semifinals .bracket-match');
        const finalMatch = document.querySelector('.bracket-round.finals .bracket-match');
        
        const qfToSfConnector = document.querySelector('.bracket-connectors.qf-to-sf');
        const sfToFConnector = document.querySelector('.bracket-connectors.sf-to-f');

        if (qfMatches.length >= 4 && sfMatches.length >= 2) {
            this.calculateQFToSFConnections(qfMatches, sfMatches, qfToSfConnector);
        }

        if (sfMatches.length >= 2 && finalMatch) {
            this.calculateSFToFinalConnections(sfMatches, finalMatch, sfToFConnector);
        }

        // Highlight winner path if tournament is complete
        if (eliminationResults.champion) {
            this.highlightWinnerPath(eliminationResults);
        }
    }

    calculateQFToSFConnections(qfMatches, sfMatches, connector) {
        // Get positions of QF matches and SF matches
        const qfPositions = Array.from(qfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = match.closest('.tournament-bracket-container').getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                right: rect.right - containerRect.left,
                element: match
            };
        });

        const sfPositions = Array.from(sfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = match.closest('.tournament-bracket-container').getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                left: rect.left - containerRect.left,
                element: match
            };
        });

        // Calculate connection line positions
        if (connector && qfPositions.length >= 4 && sfPositions.length >= 2) {
            // Position connector between QF and SF
            const connectorRect = connector.getBoundingClientRect();
            const containerRect = connector.closest('.tournament-bracket-container').getBoundingClientRect();
            
            // Calculate average positions for top and bottom bracket halves
            const topQFAvg = (qfPositions[0].top + qfPositions[1].top) / 2;
            const bottomQFAvg = (qfPositions[2].top + qfPositions[3].top) / 2;
            
            const sf1Top = sfPositions[0].top;
            const sf2Top = sfPositions[1].top;

            // Adjust connector height based on bracket spread
            const totalHeight = Math.abs(bottomQFAvg - topQFAvg) + 40;
            const centerY = (topQFAvg + bottomQFAvg) / 2;

            // Update CSS custom properties for dynamic positioning
            connector.style.setProperty('--connector-height', `${totalHeight}px`);
            connector.style.setProperty('--connector-top', `${centerY - totalHeight/2}px`);
        }
    }

    calculateSFToFinalConnections(sfMatches, finalMatch, connector) {
        // Get positions of SF matches and Final match
        const sfPositions = Array.from(sfMatches).map(match => {
            const rect = match.getBoundingClientRect();
            const containerRect = match.closest('.tournament-bracket-container').getBoundingClientRect();
            return {
                top: rect.top - containerRect.top + rect.height / 2,
                right: rect.right - containerRect.left
            };
        });

        const finalRect = finalMatch.getBoundingClientRect();
        const containerRect = finalMatch.closest('.tournament-bracket-container').getBoundingClientRect();
        const finalPosition = {
            top: finalRect.top - containerRect.top + finalRect.height / 2,
            left: finalRect.left - containerRect.left
        };

        if (connector && sfPositions.length >= 2) {
            // Calculate connection positioning for SF to Final
            const sf1Top = sfPositions[0].top;
            const sf2Top = sfPositions[1].top;
            const finalTop = finalPosition.top;

            const totalHeight = Math.abs(sf2Top - sf1Top) + 20;
            const centerY = (sf1Top + sf2Top) / 2;

            // Update CSS custom properties for SF to Final connections
            connector.style.setProperty('--connector-height', `${totalHeight}px`);
            connector.style.setProperty('--connector-top', `${centerY - totalHeight/2}px`);
        }
    }

    highlightWinnerPath(eliminationResults) {
        // Highlight the complete winner path from QF to Champion
        const qfToSfConnector = document.querySelector('.bracket-connectors.qf-to-sf');
        const sfToFConnector = document.querySelector('.bracket-connectors.sf-to-f');

        // Add active path class to highlight winner's route
        if (qfToSfConnector) {
            qfToSfConnector.classList.add('active-path');
        }
        if (sfToFConnector) {
            sfToFConnector.classList.add('active-path');
        }

        // Add smooth transition animation for winner path
        setTimeout(() => {
            const allConnectors = document.querySelectorAll('.bracket-connectors');
            allConnectors.forEach(connector => {
                connector.style.transition = 'all 0.8s ease-in-out';
            });
        }, 200);
    }

    // Responsive connection adjustments
    adjustConnectionsForScreenSize() {
        const screenWidth = window.innerWidth;
        const connectors = document.querySelectorAll('.bracket-connectors');

        connectors.forEach(connector => {
            if (screenWidth <= 768) {
                // Mobile: Hide connections, use arrows instead
                connector.style.display = 'none';
            } else if (screenWidth <= 1200) {
                // Tablet: Adjust connection sizes
                connector.classList.add('tablet-size');
            } else {
                // Desktop: Full size connections
                connector.classList.remove('tablet-size');
                connector.style.display = 'flex';
            }
        });
    }



    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        const button = document.getElementById('runSimulation');

        if (show) {
            spinner.style.display = 'block';
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        } else {
            spinner.style.display = 'none';
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play"></i> Run Simulation';
        }
    }

    showError(message) {
        const resultsArea = document.getElementById('resultsArea');
        resultsArea.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
        resultsArea.style.display = 'block';

        // Hide other result areas
        document.getElementById('singleResults').style.display = 'none';
        document.getElementById('multipleResults').style.display = 'none';
    }

    async loadTeamRankings() {
        try {
            const response = await fetch('/api/team_rankings');
            const data = await response.json();

            if (data.success) {
                this.displayTeamRankings(data.team_rankings);
            }
        } catch (error) {
            console.error('Error loading team rankings:', error);
        }
    }

    displayTeamRankings(rankings) {
        const container = document.getElementById('teamRankings');
        container.innerHTML = '';

        // Create rankings table
        const table = document.createElement('table');
        table.className = 'table table-sm table-striped';

        // Table header
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Team</th>
                    <th>Region</th>
                    <th>Power Rating</th>
                </tr>
            </thead>
            <tbody id="rankingsTableBody">
            </tbody>
        `;

        const tbody = table.querySelector('#rankingsTableBody');

        rankings.forEach(team => {
            const row = document.createElement('tr');

            // Add rank styling
            let rankClass = '';
            if (team.rank <= 3) rankClass = 'table-warning';
            else if (team.rank <= 8) rankClass = 'table-info';

            row.className = rankClass;
            row.innerHTML = `
                <td><strong>#${team.rank}</strong></td>
                <td>${team.team}</td>
                <td><span class="badge bg-secondary">${team.region}</span></td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress me-2" style="width: 100px; height: 20px;">
                            <div class="progress-bar" style="width: ${team.percentage}%"></div>
                        </div>
                        <small>${team.percentage.toFixed(1)}%</small>
                    </div>
                </td>
            `;

            tbody.appendChild(row);
        });

        container.appendChild(table);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TournamentSimulator();
});