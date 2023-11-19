module Mag_Avalanche

using Random
using ShiftedArrays
using Plots
using DataStructures

# using Infiltrator

struct IdNextNeighborCnt
	neighborCnt::Int64;
	id::Base.RefValue{Int64};
	hVal::Base.RefValue{Int64};
end

function magAvalanche( sz::Int64; varH, nDim::Int64 = 2, startPos = false )
	hLocOrder = Base.Reverse;
	hAppl = -4;
	mVal = -1;
	isSortRev = true;
	hLocIsTriggerFun = ( h -> h >= 0 );
	spinStart = false;
	if startPos
		hLocOrder = Base.Forward;
		hAppl = abs(hAppl);
		mVal = abs(mVal);
		isSortRev = false;
		hLocIsTriggerFun = ( h -> h <= 0 );
		spinStart = true;
	end
	szLst = ntuple( x->sz, nDim );
	spinArr = fill( spinStart, szLst );
	spinArrSh = [ ShiftedArrays.circshift( spinArr, ntuple( ii -> ii == iDim ? (-1)^iSh : 0, nDim ) ) for iDim = 1:nDim, iSh = 1:2 ];
	hLocArr = randn( Float64, szLst );
	hLocArr .*= varH;
	idLst = CartesianIndices( spinArr );
	lnId = length( idLst );
	idLstSh = [ ShiftedArrays.circshift( idLst, ntuple( ii -> ii == iDim ? (-1)^iSh : 0, nDim ) ) for iDim = 1 : nDim, iSh = 1 : 2 ];
	
	# hAppl = -4;
	# hHistLst = Vector{Float64}(hVal,1);
	hHistLst = fill( Float64(hAppl), 1 );
	# mVal = -1;
	mHistLst = fill( Float64(mVal), 1 );
	
	idLstSorted = Vector{CartesianIndex{nDim}}(undef,lnId);
	for ii in 1:lnId
		idLstSorted[ii] = idLst[ii];
	end
	# @infiltrate
	sort!( idLstSorted; rev = isSortRev, by = ( x -> hLocArr[x] ) );
	
	lnNextId = 2*nDim + 1;
	nextCntIdLst = [1:lnNextId;];
	nextUpIdLst = ones(Int64, lnNextId);
	nextUpHLst = zeros(lnNextId);
	neighborHAddLst = [-2*nDim:2:2*nDim;];
	 
	for iNext = 1 : lnNextId
		nextUpHLst[iNext] = hLocArr[idLstSorted[nextUpIdLst[iNext]]] + neighborHAddLst[iNext];
	end
	
	nextCntIdHeap = BinaryHeap( Base.By( ii -> nextUpHLst[ii], hLocOrder ), nextCntIdLst );
	
	hLoc = 0;
	flipLst = Vector{CartesianIndex{nDim}}(undef,0);
	
	maxNeighborCnt = 1;
	while !isempty( nextCntIdHeap )
		maxNeighborId = pop!( nextCntIdHeap );
		pos = idLstSorted[ nextUpIdLst[maxNeighborId] ];
		# @infiltrate
		
		if spinArr[pos] == spinStart
			neighborCntId = 1;
			for iDim = 1 : nDim, iSh = 1:2
				neighborCntId += spinArrSh[iDim, iSh][pos];
			end
			
			if neighborCntId == maxNeighborId
				hAppl = -( hLocArr[pos] + neighborHAddLst[maxNeighborId] );
				hHistLst = push!( hHistLst, hAppl );
				push!( flipLst, pos );
				flipPos = pos;
				# @infiltrate
				while !isempty( flipLst )
					flipPos = pop!(flipLst);
					if spinArr[flipPos] != spinStart
						continue;
					end
					spinArr[flipPos] = !spinArr[flipPos];
					for iDim = 1 : nDim, iSh = 1 : 2
						posSh = idLstSh[iDim,iSh][flipPos];
						if spinArr[posSh] != spinStart
							continue;
						end
						hLocNext = hLocArr[posSh] + hAppl;
						neighborCntId = 1;
						for iDimNxt = 1 : nDim, iShNxt = 1 : nDim
							neighborCntId += spinArrSh[iDimNxt, iShNxt][posSh];
						end
						hLocNext += neighborHAddLst[neighborCntId];
						# if hLocNext >= 0
						if hLocIsTriggerFun( hLocNext )
							push!( flipLst, posSh );
						end
					end
				end
				push!( mHistLst, boolToPN( sum( spinArr ) / lnId ) );
			end
		end
		
		nextUpIdLst[maxNeighborId] += 1;
		if nextUpIdLst[maxNeighborId] <= lnId 
			nextUpHLst[maxNeighborId] = hLocArr[ idLstSorted[nextUpIdLst[maxNeighborId]] ] + neighborHAddLst[maxNeighborId];
			push!( nextCntIdHeap, maxNeighborId );
		end
	end
	# @infiltrate
	
	return mHistLst, hHistLst;
end

function boolToPN( bVal )
	return 2*bVal - 1;
end

end
