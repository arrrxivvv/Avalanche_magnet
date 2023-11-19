module Mag_Avalanche

using Random
using ShiftedArrays
using Plots
using DataStructures

# using Infiltrator

struct IdNextNeighborCnt
	neighborCnt::Int64;
	id::Base.RefValue{Int64};
	hVal::Base.RefValue{Float64};
end

struct SpinArray
	arr::Array{Bool};
	arrSh::CircShiftedArray{Bool};
	idLst::CartesianIndices;
	idLstSh::CircShiftedArray{<:CartesianIndex};
	nDim::Int64;
	
	idLstSorted::Vector{<:CartesianIndex};
	lnId::Int64;
	
	hLocArr::Array{Float};
	
	function SpinArray( sz::Int64, nDim::Int64 )
		szLst = ntuple( x->sz, nDim );
		arr = zeros( Bool, szLst );
		hLocArr = zeros( Float64, szLst );
		arrSh = generateArrShLst( arr );
		idLst = CartesianIndices( arr );
		idLstSh = generateArrShLst( idLst );
		
		lnId = length(idLst);
		idLstSorted = Vector{CartesianIndex{nDim}}(undef,lnId);
		for ii = 1 : lnId
			idLstSorted[ii] = idLst[ii];
		end
		
		new( arr, arrSh, idLst, idLstSh, nDim, idLstSorted, lnId );
	end
end

IdNextNeighborCnt(neighborCnt::Int64) = IdNextNeighborCnt( neighborCnt, Ref(1), Ref(0.0) );
IdNextNeighborCnt(neighborCnt::Int64, id::Int64) = IdNextNeighborCnt( neighborCnt, Ref(id), Ref(0.0) );

function advanceNextNeighbor!( idNextNeighbor::IdNextNeighborCnt, idLstSorted::Vector{<:CartesianIndex}, hLocArr::Array{Float64}, neighborHAddLst::Array{Float64}, idLast::Int64 )
	idNextNeighbor.id[] += 1;
	if idNextNeighbor.id[] > idLast
		return false;
	else
		refreshNextNeighborHval!( idNextNeighbor, idLstSorted, hLocArr, neighborHAddLst );
	end
	
	return true;
end

function refreshNextNeighborHval!( idNextNeighbor::IdNextNeighborCnt, idLstSorted::Vector{<:CartesianIndex}, hLocArr::Array{Float64}, neighborHAddLst::Array{Float64} )
	idNextNeighbor.hVal[] = hLocArr[idLstSorted[idNextNeighbor.id[]]] + neighborHAddLst[idNextNeighbor.neighborCnt];
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
	
	hHistLst = fill( Float64(hAppl), 1 );
	mHistLst = fill( Float64(mVal), 1 );
	
	idLstSorted = Vector{CartesianIndex{nDim}}(undef,lnId);
	for ii in 1:lnId
		idLstSorted[ii] = idLst[ii];
	end
	sort!( idLstSorted; rev = isSortRev, by = ( x -> hLocArr[x] ) );
	
	neighborHAddLst = [-2*nDim:2.0:2*nDim;];
	lnNextId = 2*nDim + 1;
	idStart = 1;
	idNextNeighborLst = [ IdNextNeighborCnt(neighborCnt,idStart) for neighborCnt = 1 : lnNextId ];
	for ii = 1 : lnNextId
		refreshNextNeighborHval!( idNextNeighborLst[ii], idLstSorted, hLocArr, neighborHAddLst );
	end
	
	nextIdHeap = BinaryHeap( Base.By( idNext -> idNext.hVal[], hLocOrder ), idNextNeighborLst );
	
	hLoc = 0;
	flipLst = Vector{CartesianIndex{nDim}}(undef,0);
	
	maxNeighborCnt = 1;
	while !isempty( nextIdHeap )
		idNeighborMax = pop!( nextIdHeap );
		pos = idLstSorted[ idNeighborMax.id[] ];
		
		if spinArr[pos] == spinStart
			neighborCntId = 1;
			for iDim = 1 : nDim, iSh = 1:2
				neighborCntId += spinArrSh[iDim, iSh][pos];
			end
			
			if neighborCntId == idNeighborMax.neighborCnt
				hAppl = -( hLocArr[pos] + neighborHAddLst[idNeighborMax.neighborCnt] );
				hHistLst = push!( hHistLst, hAppl );
				push!( flipLst, pos );
				flipPos = pos;
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
						
						if hLocIsTriggerFun( hLocNext )
							push!( flipLst, posSh );
						end
					end
				end
				push!( mHistLst, boolToPN( sum( spinArr ) / lnId ) );
			end
		end
		
		if advanceNextNeighbor!( idNeighborMax, idLstSorted, hLocArr, neighborHAddLst, lnId )
			push!( nextIdHeap, idNeighborMax );
		end
	end
	
	return mHistLst, hHistLst;
end

function calcNeighborCntId( pos::CartesianIndex, spinArrSh::Vector{<:AbstractArray{Int64}} )
	neighborCntId = 1;
	
end

function boolToPN( bVal )
	return 2*bVal - 1;
end

function generateArrShLst( arr )
	return [ ShiftedArrays.circshift( arr, ntuple( id -> id == iDim ? (-1)^iSh : 0 ) ) for iDim = 1 : ndims(arr), iSh = 1:2 ];
end

end
